import sys
import json
import argparse
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Req.config.RunConfig import OUTPUT_ROOT
from Req.tools.parse_flow import preprocess_apk, analyze_existing_dir, cleanup_intermediate_files
from Req.llm.get_requirements import get_requirements
from Req.llm.to_test_json import save_test_json
from Req.filters.llm_requirement_filter import filter_automatable_requirements_llm
from Req.filters.hallucination_filter import verify_requirements_grounding
from Req.experiment.GetTestChain_NoFill import (
    ensure_dir,
    extract_app_intro,
    _save_json,
)
from Req.experiment.GetTestChain_NoFill import generate_tests_for_requirements


EXPERIMENT_OUTPUT = Path(__file__).resolve().parent / 'output'
LLM_LOG_DIR = Path(__file__).resolve().parent / 'llm_logs'

AVAILABLE_COMBOS = [
    '仅使用应用介绍',
    '仅使用Activity分析',
    '应用介绍+Activity分析',
    '仅使用代码解析文档',
    '代码解析文档+Activity分析',
    '应用介绍+Activity分析+代码解析文档',
]

SELECTED_COMBO = None
PER_REQ_COUNT = None


def _read_analysis(analysis_json_path: str) -> list:
    try:
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def _read_code_doc(app_name: str) -> str | None:
    base = Path(__file__).resolve().parent / 'analyzeCode' / 'test'
    mapping = {
        '桌上习惯': base / '桌上习惯deepwiki分析.md',
        'mhabit': base / '桌上习惯deepwiki分析.md',
    }
    direct = mapping.get(app_name)
    if direct:
        try:
            if direct.exists():
                return direct.read_text(encoding='utf-8')
        except Exception:
            return None

    def norm(s: str) -> str:
        s = (s or '').strip().lower()
        return re.sub(r'[^0-9a-z\u4e00-\u9fa5]+', '', s)

    key = norm(app_name)
    if not key:
        return None

    try:
        candidates = list(base.glob('*.md'))
    except Exception:
        candidates = []

    best = None
    for fp in candidates:
        name = fp.name
        if name.endswith('_deepwiki分析.md'):
            stem = name[:-len('_deepwiki分析.md')]
        elif name.endswith('deepwiki分析.md'):
            stem = name[:-len('deepwiki分析.md')]
        else:
            continue
        if norm(stem) == key:
            best = fp
            break

    if best is None:
        return None
    try:
        return best.read_text(encoding='utf-8')
    except Exception:
        return None


def _find_app_list(app_dir: Path) -> Path | None:
    candidates = [
        Path(OUTPUT_ROOT) / 'output' / 'app_list.txt',
        app_dir.parent / 'app_list.txt',
        app_dir.parent.parent / 'app_list.txt',
        Path(OUTPUT_ROOT) / 'output' / 'F-droid' / 'app_list.txt',
    ]
    for p in candidates:
        try:
            if p and p.exists():
                return p
        except Exception:
            continue
    return None


def build_prompt(app_name: str, package_name: str, main_activity: str, analysis_json_path: str, app_intro: str | None, code_doc: str | None, use_intro: bool, use_activity: bool, use_code: bool, lang: str = 'zh') -> str:
    analysis = _read_analysis(analysis_json_path) if use_activity else []
    
    if lang == 'en':
        parts = [f"App Name: {app_name}", f"Package: {package_name}", f"Launch Activity: {main_activity}"]
        parts.append("Comprehensive App Info:")
        if use_intro:
            parts.append(app_intro.strip() if app_intro and app_intro.strip() else "(No App Introduction)")
        if use_code:
            parts.append(code_doc.strip() if code_doc and code_doc.strip() else "(No Code Analysis Document)")
        if use_activity:
            for item in analysis:
                parts.append(f"- {item.get('activity')}: {item.get('function')}")
        parts.extend([
            "Generation Strategy:",
            "- Deeply understand the information above to extract core functions, key flows, and boundary scenarios.",
            "- Only generate executable scenarios within the app pages, avoiding system-level or cross-app operations.",
            "- Do not use specific Activity names; focus on user perspective and page interaction.",
            "- Cover main flows, error inputs, and exception states, generating realistic example data (in English).",
        ])
    else:
        parts = [f"应用名: {app_name}", f"包名: {package_name}", f"启动 Activity: {main_activity}"]
        parts.append("综合应用信息:")
        if use_intro:
            parts.append(app_intro.strip() if app_intro and app_intro.strip() else "（暂无应用介绍）")
        if use_code:
            parts.append(code_doc.strip() if code_doc and code_doc.strip() else "（暂无代码解析文档）")
        if use_activity:
            for item in analysis:
                parts.append(f"- {item.get('activity')}: {item.get('function')}")
        parts.extend([
            "生成策略:",
            "- 综合以上提供的信息进行深度理解，提炼核心功能、关键流程与边界场景",
            "- 仅生成可在应用页面内完成的可执行场景，避免系统级或跨应用操作",
            "- 不出现具体的 Activity 名称，聚焦用户视角与页面交互",
            "- 覆盖主流程、错误输入与异常状态，生成真实示例数据（中文）",
        ])
    return "\n".join(parts)


def _map_display_app_name(name: str) -> str:
    n = (name or '').strip().lower()
    if n in ['mhabit', 'mhhabit', '桌上习惯']:
        return '桌上习惯'
    return name


def _safe_filename(name: str) -> str:
    s = (name or '').strip()
    s = ''.join('_' if ch in '<>:"/\\|?*' else ch for ch in s)
    s = s.replace('\n', '_').replace('\r', '_').strip(' .')
    return s[:120] if len(s) > 120 else s


def _make_log_file(app_name: str) -> Path:
    LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return LLM_LOG_DIR / f"{_safe_filename(app_name)}_{ts}.jsonl"


def _log_prompt(log_file: Path, app_name: str, combo_label: str, package: str, main_activity: str, prompt: str):
    LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    record = {
        "ts": ts,
        "app": app_name,
        "combo": combo_label,
        "package": package,
        "main_activity": main_activity,
        "prompt": prompt,
    }
    with open(log_file.as_posix(), 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_existing_app_combo(app_dir: Path, combo_label: str, parsed: dict | None = None, log_file: Path | None = None, per_req_count: int | None = None, lang: str = 'zh', code_doc_override: str | None = None) -> dict:
    result = {"app_dir": app_dir.as_posix(), "ok": False, "message": "", "combo": combo_label}
    parsed = parsed or analyze_existing_dir(app_dir.as_posix())
    if not parsed.get("ok"):
        result["message"] = parsed.get("message", "解析失败")
        return result

    package = parsed.get("package", "")
    main_activity = parsed.get("main_activity", "")
    app_name = _map_display_app_name(app_dir.name)
    analysis_json_path = parsed.get("analysis_json_path")
    app_intro_text = None
    app_list_path = _find_app_list(app_dir)
    if app_list_path is not None:
        try:
            app_intro_text = extract_app_intro(app_list_path.as_posix(), package, app_dir.name)
        except Exception:
            app_intro_text = None

    code_doc = code_doc_override if code_doc_override else _read_code_doc(app_name)

    if combo_label == '仅使用应用介绍':
        use_intro, use_activity, use_code = True, False, False
    elif combo_label == '仅使用Activity分析':
        use_intro, use_activity, use_code = False, True, False
    elif combo_label == '应用介绍+Activity分析':
        use_intro, use_activity, use_code = True, True, False
    elif combo_label == '仅使用代码解析文档':
        use_intro, use_activity, use_code = False, False, True
    elif combo_label == '代码解析文档+Activity分析':
        use_intro, use_activity, use_code = False, True, True
    else:
        use_intro, use_activity, use_code = True, True, True
    prompt = build_prompt(app_name, package, main_activity, analysis_json_path, app_intro_text, code_doc, use_intro, use_activity, use_code, lang=lang)
    try:
        if log_file is not None:
            _log_prompt(log_file, app_name, combo_label, package, main_activity, prompt)
    except Exception:
        pass

    req = get_requirements(prompt, lang=lang)
    srs = (req.get('software_requirements', []) if req else [])

    # Prepare analysis data for test generation if the combo allows it
    test_gen_analysis = []
    if use_activity and analysis_json_path:
        test_gen_analysis = _read_analysis(analysis_json_path)

    # Filter SRS for hallucinations if activity analysis is available
    # if test_gen_analysis and srs:
    #     srs = verify_requirements_grounding(srs, test_gen_analysis, app_intro_text, lang=lang)

    filtered_srs = filter_automatable_requirements_llm(srs)
    
    tests = generate_tests_for_requirements(app_name, package, main_activity, filtered_srs, app_intro_text, per_req_count=per_req_count, activity_analysis=test_gen_analysis, lang=lang)

    combo_dir = EXPERIMENT_OUTPUT / combo_label / app_name
    ensure_dir(combo_dir)

    srs_path = combo_dir / f"{app_name}_SRS.json"
    _save_json(srs_path, srs)

    test_req_path = combo_dir / f"{app_name}_TestRequirements.json"
    _save_json(test_req_path, tests)

    test_json_path = save_test_json(app_name, tests, package, main_activity)
    try:
        target_test_json = combo_dir / Path(test_json_path).name
        with open(test_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _save_json(target_test_json, data)
    except Exception:
        pass

    result.update({
        "ok": True,
        "app": app_name,
        "srs": srs_path.as_posix(),
        "tests": test_req_path.as_posix(),
        "test_json": (combo_dir / Path(test_json_path).name).as_posix(),
    })
    return result


def run_for_input(args):
    combos = AVAILABLE_COMBOS
    output = []

    # 准备代码分析内容
    code_doc_content = None
    if args.source_code:
        if Path(args.source_code).exists():
            print(f"检测到源代码输入: {args.source_code}，开始运行分析流程...")
            code_doc_content = run_source_analysis(args.source_code, str(EXPERIMENT_OUTPUT))
            if not code_doc_content:
                print("源代码分析未生成有效文档。")
        else:
            print(f"提供的源代码路径不存在: {args.source_code}")
    
    if not code_doc_content and args.code_doc:
        if Path(args.code_doc).exists():
            code_doc_content = Path(args.code_doc).read_text(encoding='utf-8')

    if args.existing_dir:
        app_dir = Path(args.existing_dir).resolve()
        parsed = analyze_existing_dir(app_dir.as_posix())
        if not parsed.get("ok"):
            return [{"ok": False, "message": parsed.get("message", "解析失败")}]
        app_name = _map_display_app_name(app_dir.name)
        log_file = _make_log_file(app_name)
        if args.combo:
            output.append(process_existing_app_combo(app_dir, args.combo, parsed, log_file, args.per_req_count, code_doc_override=code_doc_content))
        else:
            for label in combos:
                output.append(process_existing_app_combo(app_dir, label, parsed, log_file, args.per_req_count, code_doc_override=code_doc_content))
        
        if CLEANUP_ENABLED:
            try:
                cleanup_intermediate_files(app_dir.as_posix())
                for r in output:
                    r['cleanup'] = True
            except Exception:
                pass

        return output

    if args.apk:
        pp = preprocess_apk(args.apk, str(OUTPUT_ROOT))
        if not pp.get("ok"):
            return [{"ok": False, "message": pp.get("message", "预处理失败")}]
        app_dir = Path(pp.get("app_dir", OUTPUT_ROOT))
        parsed = analyze_existing_dir(app_dir.as_posix())
        if not parsed.get("ok"):
            return [{"ok": False, "message": parsed.get("message", "解析失败")}]
        app_name = _map_display_app_name(app_dir.name)
        log_file = _make_log_file(app_name)
        if args.combo:
            output.append(process_existing_app_combo(app_dir, args.combo, parsed, log_file, args.per_req_count, code_doc_override=code_doc_content))
        else:
            for label in combos:
                output.append(process_existing_app_combo(app_dir, label, parsed, log_file, args.per_req_count, code_doc_override=code_doc_content))
        
        if CLEANUP_ENABLED:
            try:
                cleanup_intermediate_files(app_dir.as_posix())
                for r in output:
                    r['cleanup'] = True
            except Exception:
                pass
                
        return output

    return [{"ok": False, "message": "未提供输入"}]


CLEANUP_ENABLED = False


def _process_app_all_combos(app_dir: Path) -> list:
    combos = [SELECTED_COMBO] if SELECTED_COMBO else [
        '仅使用应用介绍',
        '仅使用Activity分析',
        '应用介绍+Activity分析',
        '仅使用代码解析文档',
        '代码解析文档+Activity分析',
        '应用介绍+Activity分析+代码解析文档'
    ]
    parsed = analyze_existing_dir(app_dir.as_posix())
    if not parsed.get("ok"):
        return [{"app_dir": app_dir.as_posix(), "ok": False, "message": parsed.get("message", "解析失败"), "combo": label} for label in combos]
    app_name = _map_display_app_name(app_dir.name)
    log_file = _make_log_file(app_name)
    results = []
    for label in combos:
        try:
            results.append(process_existing_app_combo(app_dir, label, parsed, log_file, PER_REQ_COUNT))
        except Exception as e:
            results.append({"app_dir": app_dir.as_posix(), "ok": False, "message": str(e), "combo": label})
            
    if CLEANUP_ENABLED:
        try:
            cleanup_intermediate_files(app_dir.as_posix())
            # Mark cleanup in results
            for r in results:
                r['cleanup'] = True
        except Exception as e:
            pass
            
    return results


def run_for_batch_dir(batch_dir: str, threads: int) -> list:
    base = Path(batch_dir).resolve()
    if not base.exists():
        return [{"ok": False, "message": f"目录不存在: {base.as_posix()}"}]
    app_dirs = [p for p in base.iterdir() if p.is_dir()]
    results = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        fut_map = {ex.submit(_process_app_all_combos, d): d for d in app_dirs}
        for fut in as_completed(fut_map):
            try:
                res = fut.result()
                results.extend(res)
            except Exception as e:
                d = fut_map[fut]
                results.append({"app_dir": d.as_posix(), "ok": False, "message": str(e)})
    return results


def main():
    parser = argparse.ArgumentParser(description='综合理解：三信息六组合提示词生成')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--apk', help='APK 文件路径')
    group.add_argument('--existing_dir', help='已反编译的应用目录')
    parser.add_argument('--batch_dir', help='批量处理的目录（并发处理其下所有应用子目录）')
    parser.add_argument('--threads', type=int, default=4, help='并发线程数')
    parser.add_argument('--code_doc', type=str, default=None, help='代码分析文档路径')
    parser.add_argument('--source_code', type=str, default=None, help='源代码目录路径 (如果提供，将自动运行代码分析)')
    parser.add_argument('--combo', choices=AVAILABLE_COMBOS, help='仅跑指定组合')
    parser.add_argument('--per_req_count', type=int, default=None, help='每条功能需求生成测试场景条数（不传则由模型自行决定）')
    parser.add_argument('--cleanup', action='store_true', help='任务完成后清理反编译中间文件以节省空间')
    args = parser.parse_args()

    ensure_dir(EXPERIMENT_OUTPUT)
    global SELECTED_COMBO
    SELECTED_COMBO = args.combo
    global PER_REQ_COUNT
    PER_REQ_COUNT = args.per_req_count
    global CLEANUP_ENABLED
    CLEANUP_ENABLED = args.cleanup
    if args.batch_dir:
        results = run_for_batch_dir(args.batch_dir, args.threads)
    else:
        results = run_for_input(args)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
