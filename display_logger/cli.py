import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from .config import Box, DecimalPoint, Profile
from .records import JsonlWriter, make_record, to_csv


def positive_float(text):
    number = float(text)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Use um número positivo e finito.")
    return number


def positive_int(text):
    number = int(text)
    if number <= 0:
        raise argparse.ArgumentTypeError("Use um inteiro positivo.")
    return number


def require_gui():
    if sys.platform.startswith("linux") and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
        raise ValueError("Sem sessão gráfica. Calibre em um computador com tela; use run --no-preview para capturar sem janela.")


def open_capture(args):
    import cv2

    is_video = bool(getattr(args, "video", None))
    source = str(args.video) if is_video else args.camera
    requested = getattr(args, "camera_backend", "auto")
    if is_video:
        backends = [cv2.CAP_ANY]
    elif requested == "v4l2":
        backends = [cv2.CAP_V4L2]
    elif requested == "gstreamer":
        backends = [cv2.CAP_GSTREAMER]
    elif sys.platform.startswith("linux"):
        # Evita selecionar GStreamer primeiro para webcams locais no Linux.
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    for index, backend in enumerate(backends):
        capture = cv2.VideoCapture(source, backend)
        if not capture.isOpened():
            capture.release()
            if index + 1 < len(backends):
                print("V4L2 não abriu a webcam; tentando seleção automática do OpenCV.",
                      file=sys.stderr, flush=True)
            continue
        try:
            if not is_video:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                backend_name = capture.getBackendName()
                # GStreamer não implementa essa propriedade e emite avisos.
                if backend_name == "V4L2":
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"Webcam {source} aberta usando {backend_name}.", flush=True)
            return capture
        except Exception:
            capture.release()
            raise
    raise ValueError(f"Não foi possível abrir a câmera/vídeo {source!r} (backend: {requested}).")


def load_image(path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Não foi possível ler a imagem: {path}.")
    return image


def load_tracker(args, profile):
    if not profile.track_display or getattr(args, "no_tracking", False):
        return None
    from .tracking import DisplayTracker

    reference_path = Path(profile.reference_image)
    if not reference_path.is_absolute():
        reference_path = Path(args.profile).parent / reference_path
    return DisplayTracker(profile, load_image(reference_path))


def load_reader(args, profile):
    from .vision import DisplayReader

    weights = None
    if profile.engine == "yolo":
        weights = Path(profile.yolo_weights)
        if not weights.is_absolute():
            weights = Path(args.profile).parent / weights
    return DisplayReader(profile, weights_path=weights)


def read_frame(reader, tracker, frame):
    from .vision import Reading

    if tracker is not None:
        try:
            frame = tracker.align(frame)
        except ValueError as exc:
            return frame, Reading("", None, 0, str(exc))
    return frame, reader.read(frame)


def select_box(image, title, optional=False):
    import cv2

    print(title + (" (ESC para ignorar)" if optional else ""), flush=True)
    # Amplia recortes pequenos para não exigir seleção de caixas de poucos pixels.
    scale = max(1.0, min(800 / image.shape[1], 600 / image.shape[0], 6.0))
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale > 1 else image
    rect = cv2.selectROI(title, shown, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if not rect[2] or not rect[3]:
        if optional:
            return None
        raise ValueError("Calibração cancelada; nenhuma região foi salva.")
    return Box.from_pixels(rect, shown.shape[1], shown.shape[0])


def select_perspective(frame):
    import cv2
    import numpy as np
    from .geometry import perspective_from_points

    title = "Perspectiva: selecione os 4 cantos do visor"
    scale = min(1100 / frame.shape[1], 620 / frame.shape[0], 2.0)
    shown = cv2.resize(frame, None, fx=scale, fy=scale)
    points = []
    labels = ("1 SE", "2 SD", "3 ID", "4 IE")
    header = 75

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            y -= header
            if 0 <= x < shown.shape[1] and 0 <= y < shown.shape[0]:
                points.append((x, y))

    print("Clique nos cantos do VISOR, incluindo sinal e pontos:\n"
          "1 superior esquerdo; 2 superior direito; 3 inferior direito; 4 inferior esquerdo.\n"
          "Enter confirma; Backspace desfaz; R limpa; ESC cancela.", flush=True)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, click)
    try:
        while True:
            canvas = cv2.copyMakeBorder(shown, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30,) * 3)
            next_label = labels[len(points)] if len(points) < 4 else "ENTER para confirmar"
            cv2.putText(canvas, f"Cantos: SE > SD > ID > IE | Proximo: {next_label}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,) * 3, 1)
            cv2.putText(canvas, "ENTER confirma | Backspace desfaz | R limpa | ESC cancela",
                        (10, 53), cv2.FONT_HERSHEY_SIMPLEX, .5, (230,) * 3, 1)
            if points:
                polygon = np.array([(x, y + header) for x, y in points], dtype=np.int32)
                cv2.polylines(canvas, [polygon], len(points) == 4, (0, 220, 255), 2)
                for i, (x, y) in enumerate(polygon):
                    cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
                    cv2.putText(canvas, labels[i], (int(x) + 6, int(y) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Seleção dos cantos cancelada; perfil não alterado.")
            if key in (8, 127) and points:
                points.pop()
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                pixels = [(x * (frame.shape[1] - 1) / (shown.shape[1] - 1),
                           y * (frame.shape[0] - 1) / (shown.shape[0] - 1)) for x, y in points]
                try:
                    return perspective_from_points(pixels, frame.shape[1], frame.shape[0])
                except ValueError as exc:
                    print(f"{exc} Use R para refazer os cantos.", flush=True)
    finally:
        cv2.destroyWindow(title)


def reference_frame(args):
    import cv2

    if args.image:
        return load_image(args.image)
    capture = open_capture(args)
    title = "Calibracao: ESPACO congela; Q sai"
    print("A imagem está ao vivo. Clique na janela da webcam e pressione ESPAÇO para congelar.\n"
          "Depois, selecione as regiões com o mouse e confirme cada uma com Enter. Q/ESC encerra.",
          flush=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Não foi possível obter um quadro da webcam.")
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                return frame
            if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Calibração cancelada.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


def calibrate(args):
    import cv2

    base = Profile.load(args.from_profile) if args.from_profile else None
    if base is not None and base.engine != args.engine and args.engine != "yolo":
        raise ValueError("O perfil de origem precisa usar o mesmo motor de reconhecimento.")
    if args.digit_slant is None:
        args.digit_slant = base.digit_slant if base else 0
    if args.otsu_offset is None:
        args.otsu_offset = base.otsu_offset if base else 0
    if args.segment_search is None:
        args.segment_search = base.segment_search if base else 0
    yolo_weights = None
    if args.engine == "yolo":
        yolo_weights = Path(args.weights or (base.yolo_weights if base else None) or "models/digits/best.pt")
        if not yolo_weights.is_absolute() and not args.weights and base and base.yolo_weights:
            yolo_weights = Path(args.from_profile).parent / yolo_weights
        yolo_weights = str(yolo_weights.resolve())
    require_gui()
    if args.engine in ("tesseract", "yolo") and (args.decimals is not None or args.no_decimal):
        raise ValueError("No OCR/YOLO, o ponto é lido da imagem; omita --decimals e --no-decimal.")
    if args.decimals is not None and not 0 <= args.decimals <= 12:
        raise ValueError("--decimals precisa estar entre 0 e 12.")
    frame = reference_frame(args)
    try:
        from .geometry import rectify

        use_perspective = args.perspective if args.perspective is not None else bool(base and base.perspective)
        perspective = select_perspective(frame) if use_perspective else None
        plane = rectify(frame, perspective)
        roi = select_box(plane, "Numeros: inclua digitos, pontos e sinal; exclua unidade e icones")
        cropped = roi.crop(plane)
        digits, points, sign = [], [], None
        decimal_mode = "detect"
        if args.engine == "seven_segment":
            print("Marque a célula INTEIRA de cada dígito, mesmo se mostrar 1 ou estiver apagado.")
            print("Inclua a altura total dos segmentos; exclua pontos decimais da célula.")
            for index in range(args.digits):
                digits.append(select_box(cropped, f"Digito {index + 1} de {args.digits}: esquerda para direita"))
            if not args.unsigned:
                sign = select_box(cropped, "Regiao exclusiva do sinal negativo, mesmo apagado", optional=True)
                if sign is None:
                    print("Sinal externo não configurado. Valores negativos exigem sinal em uma célula de dígito.")
            if args.decimals is not None:
                decimal_mode = "fixed"
            elif args.no_decimal:
                decimal_mode = "none"
            else:
                print("Marque TODOS os pontos que podem acender, mesmo apagados. Recorte cada ponto de perto.")
                for after in range(1, args.digits):
                    point = select_box(cropped, f"Ponto decimal depois do digito {after}", optional=True)
                    if point is not None:
                        points.append(DecimalPoint(after, point))
        reference_path = Path(args.profile).with_suffix(".referencia.png")
        profile = Profile(
            name=args.name or Path(args.profile).stem, unit=args.unit, roi=roi, engine=args.engine,
            digits=digits, sign=sign, decimal_mode=decimal_mode, decimal_points=points,
            decimal_places=args.decimals or 0,
            require_decimal=(not args.allow_integer) if args.engine == "yolo" else args.require_decimal,
            polarity=args.polarity, threshold=args.threshold, min_confidence=args.min_confidence,
            digit_slant=args.digit_slant, otsu_offset=args.otsu_offset,
            segment_probes=list(base.segment_probes) if base and args.engine == "seven_segment" else [],
            segment_search=args.segment_search,
            track_display=not args.no_tracking, reference_image=reference_path.name,
            perspective=perspective,
            value_min=args.value_min, value_max=args.value_max,
            yolo_weights=yolo_weights,
            yolo_expected_digits=args.digits if args.engine == "yolo" else None,
        )
        profile.save(args.profile)
        from .diagnostics import write_image
        from .vision import DisplayReader

        write_image(reference_path, frame)
        if perspective:
            corrected_path = Path(args.profile).with_suffix(".retificado.png")
            write_image(corrected_path, plane)
            print(f"Visor retificado salvo em {corrected_path}.")
        print(f"Perfil salvo em {args.profile}. Imagem de referência: {reference_path}.")
        if profile.engine == "yolo" and not Path(profile.yolo_weights).is_file():
            print("Calibração geométrica concluída. A leitura YOLO aguarda os pesos treinados de dígitos.")
            return 0
        reference_reading = load_reader(args, profile).read(frame)
        if reference_reading.valid:
            print(f"Leitura de referência: {reference_reading.raw} {profile.unit}. Confira com o visor.")
        else:
            print(f"A leitura de referência foi rejeitada: {reference_reading.reason}\n"
                  "Confira se o visor está ligado, nítido e inteiramente dentro das regiões selecionadas.")
    finally:
        cv2.destroyAllWindows()
    return 0


def draw_preview(frame, profile, reading, status):
    import cv2
    from .diagnostics import preview_image

    cv2.imshow("Display logger", preview_image(frame, profile, reading, status))


def run(args):
    import cv2
    from .diagnostics import save_diagnostics
    from .vision import DisplayReader, Reading, StabilityGate

    profile = Profile.load(args.profile)
    test_id = (args.test_id if args.test_id is not None else input("Identificação do teste: ")).strip()
    if not test_id:
        raise ValueError("A identificação do teste não pode ficar vazia.")
    if not args.no_preview:
        require_gui()
    reader, gate = load_reader(args, profile), StabilityGate(args.stable_frames)
    tracker = load_tracker(args, profile)
    capture = open_capture(args)
    count, frame_index = 0, 0
    last_diagnostic_reason = None
    diagnostic_dir = Path(args.debug_dir) if args.debug_dir else (
        Path(args.out).parent / (Path(args.out).stem + "-diagnostico"))
    started = next_record = time.monotonic()
    fps = capture.get(cv2.CAP_PROP_FPS) if args.video else None
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        fps = 30
    print(f"Teste {test_id!r}; perfil {profile.name!r}; arquivo {args.out}. Ctrl+C encerra.")
    try:
        with JsonlWriter(args.out) as log:
            while args.duration is None or time.monotonic() - started < args.duration:
                if fps is not None:
                    delay = started + frame_index / fps - time.monotonic()
                    if delay > 0:
                        time.sleep(min(delay, 1 / fps))
                ok, frame = capture.read()
                timestamp = time.time()  # Instante no computador, logo após obter o quadro.
                if not ok:
                    if args.video:
                        break
                    failure = Reading("", None, 0, "Falha de captura; webcam desconectada ou indisponível.")
                    log.write(make_record(timestamp, test_id, profile, failure, "invalid"))
                    raise ValueError(failure.reason)
                frame_index += 1
                source_frame = frame
                frame, reading = read_frame(reader, tracker, frame)
                status = gate.update(reading)
                now = time.monotonic()
                if now >= next_record:
                    if args.debug_dir and status == "invalid" and reading.reason != last_diagnostic_reason:
                        save_diagnostics(frame, profile, diagnostic_dir, reading,
                                         source_frame=source_frame if tracker else None)
                        last_diagnostic_reason = reading.reason
                    record = make_record(timestamp, test_id, profile, reading, status)
                    log.write(record)
                    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
                    count += 1
                    next_record = now + args.interval
                if args.samples is not None and count >= args.samples:
                    break
                if not args.no_preview:
                    draw_preview(frame, profile, reading, status)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        save_diagnostics(frame, profile, diagnostic_dir, reading,
                                         source_frame=source_frame if tracker else None)
                        print(f"Diagnóstico salvo em {diagnostic_dir}.", flush=True)
                    if key in (ord("q"), 27) or cv2.getWindowProperty("Display logger", cv2.WND_PROP_VISIBLE) < 1:
                        break
        print(f"{count} registros gravados em {args.out}.")
    finally:
        capture.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
    if frame_index == 0:
        raise ValueError("Nenhum quadro foi recebido.")
    return 0


def read_image(args):
    from .diagnostics import save_diagnostics
    from .vision import DisplayReader

    profile = Profile.load(args.profile)
    if not args.test_id.strip():
        raise ValueError("A identificação do teste não pode ficar vazia.")
    source_frame = load_image(args.image)
    tracker = load_tracker(args, profile)
    frame, reading = read_frame(load_reader(args, profile), tracker, source_frame)
    record = make_record(time.time(), args.test_id, profile, reading, "ok" if reading.valid else "invalid")
    if args.out:
        with JsonlWriter(args.out) as log:
            log.write(record)
    if args.debug_dir:
        save_diagnostics(frame, profile, args.debug_dir, reading,
                         source_frame=source_frame if tracker else None)
    print(json.dumps(record, ensure_ascii=False, allow_nan=False))
    return 0 if reading.valid else 2


def convert(args):
    count = to_csv(args.input, args.output, args.delimiter)
    print(f"{count} registros exportados para {args.output}.")
    return 0


def demo(args):
    from .synthetic import create_demo

    count = create_demo(args.out_dir)
    print(f"Demonstração sintética: {count} registros no total em {args.out_dir}/leituras.txt e leituras.csv.")
    return 0


def camera_options(parser, allow_image=False, allow_video=False):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Índice da webcam; padrão: 0")
    if allow_image:
        source.add_argument("--image", help="Calibrar a partir de uma foto")
    if allow_video:
        source.add_argument("--video", help="Vídeo para validar o reconhecimento; timestamp é o da execução")
    parser.add_argument("--camera-backend", choices=("auto", "v4l2", "gstreamer"), default="auto",
                        help="Driver da webcam; auto prioriza V4L2 no Linux")
    parser.add_argument("--width", type=positive_int, default=1280)
    parser.add_argument("--height", type=positive_int, default=720)


def build_parser():
    parser = argparse.ArgumentParser(description="Leitura de displays por webcam, TXT JSONL e CSV.")
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="Criar perfil selecionando regiões na imagem")
    camera_options(calibration, allow_image=True)
    calibration.add_argument("--profile", required=True)
    calibration.add_argument("--from-profile", help="Reutilizar sondas, inclinação e ajuste Otsu de um perfil existente")
    perspective_options = calibration.add_mutually_exclusive_group()
    perspective_options.add_argument("--perspective", action="store_true", default=None,
                                     help="Marcar quatro cantos e corrigir a perspectiva antes de selecionar a ROI")
    perspective_options.add_argument("--no-perspective", action="store_false", dest="perspective",
                                     help="Selecionar diretamente na câmera, sem correção de perspectiva")
    calibration.add_argument("--name")
    calibration.add_argument("--unit", required=True, help="Ex.: volts, amperes, ohms, mV")
    calibration.add_argument("--engine", choices=("seven_segment", "tesseract", "yolo"), default="seven_segment")
    calibration.add_argument("--weights", help="Pesos YOLO de dígitos; podem ser treinados depois da calibração")
    calibration.add_argument("--allow-integer", action="store_true", help="No YOLO, permitir leitura sem ponto decimal")
    calibration.add_argument("--digits", type=positive_int, default=4)
    decimal_options = calibration.add_mutually_exclusive_group()
    decimal_options.add_argument("--decimals", type=int, help="Casas decimais FIXAS; exige escala fixa no instrumento")
    decimal_options.add_argument("--no-decimal", action="store_true", help="Instrumento exibe apenas inteiros")
    calibration.add_argument("--unsigned", action="store_true", help="Omitir seleção de sinal externo")
    calibration.add_argument("--require-decimal", action="store_true")
    calibration.add_argument("--polarity", choices=("dark", "light"), default="dark")
    calibration.add_argument("--threshold", choices=("otsu", "adaptive"), default="otsu")
    calibration.add_argument("--digit-slant", type=float,
                             help="Inclinação horizontal dos segmentos como fração da largura da célula (-0.5 a 0.5)")
    calibration.add_argument("--otsu-offset", type=float,
                             help="Ajuste do limiar Otsu em níveis de cinza (-64 a 64)")
    calibration.add_argument("--segment-search", type=float,
                             help="Busca local do traço ao redor de cada sonda, como fração da célula (0 a 0.10)")
    calibration.add_argument("--no-tracking", action="store_true", help="Desativar rastreamento pela imagem de referência")
    calibration.add_argument("--min-confidence", type=float, default=.55)
    calibration.add_argument("--value-min", type=float)
    calibration.add_argument("--value-max", type=float)
    calibration.set_defaults(func=calibrate)

    capture = commands.add_parser("run", help="Ler continuamente e registrar um teste")
    camera_options(capture, allow_video=True)
    capture.add_argument("--profile", required=True)
    capture.add_argument("--test-id", help="Se omitido, será solicitado no terminal")
    capture.add_argument("--out", default="logs/leituras.txt")
    capture.add_argument("--interval", type=positive_float, default=1, help="Intervalo mínimo entre registros, em segundos")
    capture.add_argument("--stable-frames", type=positive_int, default=3)
    capture.add_argument("--duration", type=positive_float, help="Duração máxima aproximada, em segundos")
    capture.add_argument("--samples", type=positive_int, help="Máximo de registros, incluindo falhas")
    capture.add_argument("--no-preview", action="store_true")
    capture.add_argument("--no-tracking", action="store_true")
    capture.add_argument("--debug-dir", help="Salvar imagens e sondas quando o motivo de falha mudar; S salva manualmente")
    capture.set_defaults(func=run)

    image = commands.add_parser("read-image", help="Validar um único quadro, sem confirmação temporal")
    image.add_argument("--profile", required=True)
    image.add_argument("--image", required=True)
    image.add_argument("--test-id", default="imagem-de-validacao")
    image.add_argument("--out")
    image.add_argument("--debug-dir")
    image.add_argument("--no-tracking", action="store_true")
    image.set_defaults(func=read_image)

    csv_parser = commands.add_parser("csv", help="Converter TXT JSONL em CSV")
    csv_parser.add_argument("--input", required=True)
    csv_parser.add_argument("--output", required=True)
    csv_parser.add_argument("--delimiter", default=",")
    csv_parser.set_defaults(func=convert)

    demonstration = commands.add_parser("demo", help="Gerar imagens e logs sintéticos, sem câmera")
    demonstration.add_argument("--out-dir", default="demo")
    demonstration.set_defaults(func=demo)
    from .yolo_workflow import add_commands
    add_commands(commands)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nCaptura encerrada.", file=sys.stderr)
        return 130
    except (OSError, ValueError, EOFError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
