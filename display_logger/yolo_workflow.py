"""Comandos de coleta, revisão, treino e avaliação do leitor YOLO."""

from dataclasses import replace
import json
import math
from pathlib import Path
import time

import cv2

from .config import Profile
from .geometry import display_image
from .yolo_data import dataset_status, prepare_dataset, save_sample


def collect(args):
    from .cli import load_tracker, open_capture, require_gui

    profile = Profile.load(args.profile)
    tracker = load_tracker(args, profile)
    if not args.no_preview:
        require_gui()
    capture = open_capture(args)
    count, received = 0, 0
    started = time.monotonic()
    next_sample = 0.
    fps = capture.get(cv2.CAP_PROP_FPS) if args.video else None
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        fps = 30.
    try:
        while count < args.samples and time.monotonic() - started < args.duration:
            ok, frame = capture.read()
            if not ok:
                if args.video:
                    break
                raise ValueError("A webcam deixou de fornecer imagens.")
            received += 1
            if not args.video and received <= 8:
                continue
            sample_time = (received - 1) / fps if fps is not None else time.monotonic() - started
            if fps is not None and sample_time > args.duration:
                break
            if sample_time < next_sample and (args.no_preview or args.video):
                continue
            aligned = tracker.align(frame) if tracker else frame
            roi = display_image(aligned, profile)
            if sample_time >= next_sample:
                target = save_sample(args.dataset, args.session, roi,
                                     {"source_profile": str(Path(args.profile).resolve()),
                                      "source": args.video or f"camera:{args.camera}",
                                      "source_seconds": sample_time if fps is not None else None})
                count += 1
                next_sample = sample_time + args.interval
                print(f"[{count}/{args.samples}] {target}", flush=True)
            if not args.no_preview:
                enlarged = cv2.resize(roi, None, fx=max(1., 700 / roi.shape[1]), fy=max(1., 700 / roi.shape[1]))
                cv2.imshow("Coleta YOLO - Q encerra", enlarged)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
    if count == 0:
        raise ValueError("Nenhuma amostra capturada.")
    print(f"{count} imagens SEM rótulos. Revise com yolo-annotate antes de treinar.")
    return 0


def import_images(args):
    from .cli import load_image, load_tracker

    profile = Profile.load(args.profile) if args.profile else None
    tracker = load_tracker(args, profile) if profile else None
    count = 0
    for source in args.images:
        frame = load_image(source)
        if tracker:
            frame = tracker.align(frame)
        roi = display_image(frame, profile) if profile else frame
        target = save_sample(args.dataset, args.session, roi, {"source_file": str(Path(source).resolve())})
        print(target)
        count += 1
    print(f"{count} imagens importadas sem rótulos.")
    return 0


def annotate(args):
    from .cli import require_gui
    from .yolo_annotator import annotate_session

    require_gui()
    return annotate_session(args.dataset, args.session)


def status(args):
    print(json.dumps(dataset_status(args.dataset), ensure_ascii=False, indent=2))
    return 0


def prepare(args):
    manifest = prepare_dataset(args.dataset, args.output, args.val_session, args.test_session, args.seed)
    print(f"Dataset salvo em {args.output}/data.yaml. Sessões: {manifest['sessions']}")
    if manifest["missing_train_classes"]:
        print("Treino ainda bloqueado: faltam exemplos de " + ", ".join(manifest["missing_train_classes"]))
    return 0


def train(args):
    from .yolo_training import train_model

    target = train_model(args)
    print(f"Pesos treinados: {target}. Avalie com yolo-evaluate antes de usar na bancada.")
    return 0


def evaluate(args):
    from .yolo_training import evaluate_model

    result = evaluate_model(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def make_profile(args):
    source, target = Path(args.from_profile), Path(args.profile)
    if source.resolve() == target.resolve():
        raise ValueError("Salve o motor YOLO em outro perfil, preservando a calibração de origem.")
    base = Profile.load(source)
    reference = base.reference_image
    if reference and not Path(reference).is_absolute():
        reference = str((source.parent / reference).resolve())
    profile = replace(base, name=target.stem, engine="yolo", digits=[], decimal_points=[], sign=None,
                      segment_probes=[], segment_search=0, digit_slant=0, decimal_mode="detect",
                      require_decimal=not args.allow_integer, reference_image=reference,
                      yolo_weights=str(Path(args.weights).resolve()), yolo_device=args.device,
                      yolo_imgsz=args.imgsz, yolo_expected_digits=(args.expected_digits or
                                                                base.yolo_expected_digits or len(base.digits) or None))
    profile.save(target)
    print(f"Perfil YOLO salvo em {target}.")
    if not Path(profile.yolo_weights).is_file():
        print("Pesos ainda ausentes. Este perfil só poderá ler após coleta, anotação e treinamento.")
    return 0


def add_commands(commands):
    from .cli import camera_options, positive_float, positive_int

    capture = commands.add_parser("yolo-collect", help="Coletar recortes do visor para anotar")
    camera_options(capture, allow_video=True)
    capture.add_argument("--profile", required=True)
    capture.add_argument("--dataset", default="datasets/digits")
    capture.add_argument("--session", required=True)
    capture.add_argument("--samples", type=positive_int, default=50)
    capture.add_argument("--interval", type=positive_float, default=1)
    capture.add_argument("--duration", type=positive_float, default=60)
    capture.add_argument("--no-preview", action="store_true")
    capture.add_argument("--no-tracking", action="store_true")
    capture.set_defaults(func=collect)

    incoming = commands.add_parser("yolo-import", help="Importar imagens sem rótulos; --profile recorta quadros completos")
    incoming.add_argument("--images", nargs="+", required=True)
    incoming.add_argument("--profile")
    incoming.add_argument("--dataset", default="datasets/digits")
    incoming.add_argument("--session", required=True)
    incoming.add_argument("--no-tracking", action="store_true")
    incoming.set_defaults(func=import_images)

    annotation = commands.add_parser("yolo-annotate", help="Marcar dígitos, ponto e sinal com o mouse")
    annotation.add_argument("--dataset", default="datasets/digits")
    annotation.add_argument("--session", required=True)
    annotation.set_defaults(func=annotate)

    stats = commands.add_parser("yolo-status", help="Ver rótulos revisados, pendências e classes faltantes")
    stats.add_argument("--dataset", default="datasets/digits")
    stats.set_defaults(func=status)

    split = commands.add_parser("yolo-prepare", help="Separar sessões revisadas entre treino, validação e teste")
    split.add_argument("--dataset", default="datasets/digits")
    split.add_argument("--output", default="datasets/digits/prepared")
    split.add_argument("--val-session")
    split.add_argument("--test-session")
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=prepare)

    training = commands.add_parser("yolo-train", help="Treinar um detector de caracteres com dados revisados")
    training.add_argument("--data", default="datasets/digits/prepared/data.yaml")
    training.add_argument("--model", default="yolo26n.pt")
    training.add_argument("--epochs", type=positive_int, default=100)
    training.add_argument("--imgsz", type=positive_int, default=640)
    training.add_argument("--batch", type=positive_int, default=8)
    training.add_argument("--device", default="cpu")
    training.add_argument("--project", default="runs/digits")
    training.add_argument("--output", default="models/digits/best.pt")
    training.set_defaults(func=train)

    evaluation = commands.add_parser("yolo-evaluate", help="Avaliar leitura completa nas imagens de teste")
    evaluation.add_argument("--data", default="datasets/digits/prepared/data.yaml")
    evaluation.add_argument("--weights", default="models/digits/best.pt")
    evaluation.add_argument("--output", default="reports/yolo-evaluation.json")
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--imgsz", type=positive_int, default=640)
    evaluation.add_argument("--expected-digits", type=positive_int, default=4)
    evaluation.add_argument("--allow-integer", action="store_true")
    evaluation.set_defaults(func=evaluate)

    profile = commands.add_parser("yolo-profile", help="Reutilizar ROI e perspectiva com o motor YOLO")
    profile.add_argument("--from-profile", required=True)
    profile.add_argument("--profile", default="perfis/multimetro-yolo.json")
    profile.add_argument("--weights", default="models/digits/best.pt")
    profile.add_argument("--device", default="cpu")
    profile.add_argument("--imgsz", type=positive_int, default=640)
    profile.add_argument("--expected-digits", type=positive_int)
    profile.add_argument("--allow-integer", action="store_true")
    profile.set_defaults(func=make_profile)
