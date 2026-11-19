"""Anotação manual de caracteres no recorte que será usado na inferência."""

import cv2

from .yolo_data import CLASSES, labels_path, read_labels, review_path, save_review, session_dir


def annotate_session(root, session):
    images = sorted((session_dir(root, session) / "images").glob("*.png"))
    if not images:
        raise ValueError("Esta sessão não possui imagens; use yolo-collect ou yolo-import.")
    pending = [i for i, p in enumerate(images) if not review_path(p).exists()]
    index = pending[0] if pending else 0
    title, header, active = "Anotar caracteres YOLO", 100, 0
    saved = 0
    try:
        while 0 <= index < len(images):
            source = images[index]
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError(f"Imagem inválida: {source}.")
            scale = min(1000 / frame.shape[1], 550 / frame.shape[0], 12.)
            shown = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            width, height = shown.shape[1], shown.shape[0]
            labels = read_labels(labels_path(source)) if labels_path(source).exists() else []
            state = {"start": None, "end": None, "negative": False, "active": active}

            def mouse(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN and not header <= y < header + height:
                    return
                y -= header
                x, y = min(max(x, 0), width), min(max(y, 0), height)
                if event == cv2.EVENT_LBUTTONDOWN:
                    state["start"], state["end"] = (x, y), (x, y)
                elif event == cv2.EVENT_MOUSEMOVE and state["start"] is not None:
                    state["end"] = (x, y)
                elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
                    x0, x1 = sorted((state["start"][0], x))
                    y0, y1 = sorted((state["start"][1], y))
                    if x1 - x0 >= 2 and y1 - y0 >= 2:
                        labels.append((state["active"], (x0 + x1) / (2 * width),
                                       (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height))
                        state["negative"] = False
                    state["start"] = state["end"] = None

            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(title, mouse)
            while True:
                canvas = cv2.copyMakeBorder(shown, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30,) * 3)
                lines = [f"{index + 1}/{len(images)} | Classe: {CLASSES[state['active']]} | {source.stem}",
                         "0-9: digito | .: ponto | -: sinal | arraste uma caixa por caractere",
                         "ENTER salva | B: visor invalido/apagado | R limpa | Backspace desfaz",
                         "N/P: proxima/anterior | Q/ESC sai" + (" | NEGATIVO" if state["negative"] else "")]
                for row, text in enumerate(lines):
                    cv2.putText(canvas, text, (8, 20 + row * 23), cv2.FONT_HERSHEY_SIMPLEX, .47, (240,) * 3, 1)
                for class_id, x, y, w, h in labels:
                    x0, y0 = round((x - w / 2) * width), round((y - h / 2) * height) + header
                    x1, y1 = round((x + w / 2) * width), round((y + h / 2) * height) + header
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 230, 0), 1)
                    cv2.putText(canvas, CLASSES[class_id], (x0, max(header + 12, y0 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 230, 0), 1)
                if state["start"] is not None:
                    a, b = state["start"], state["end"]
                    cv2.rectangle(canvas, (a[0], a[1] + header), (b[0], b[1] + header), (0, 200, 255), 1)
                cv2.imshow(title, canvas)
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    print(f"{saved} imagens revisadas nesta execução; as pendentes não entram no treino.")
                    return 0
                if ord("0") <= key <= ord("9"):
                    state["active"] = active = key - ord("0")
                elif key in (ord("."), ord(",")):
                    state["active"] = active = 10
                elif key == ord("-"):
                    state["active"] = active = 11
                elif key in (8, 127) and labels:
                    labels.pop()
                elif key == ord("r"):
                    labels.clear()
                    state["negative"] = False
                elif key == ord("b"):
                    labels.clear()
                    state["negative"] = True
                elif key in (ord("n"), ord("p")):
                    index += 1 if key == ord("n") else -1
                    index = max(0, index)
                    break
                elif key in (10, 13):
                    try:
                        text = save_review(source, labels, state["negative"])
                    except ValueError as exc:
                        print(f"Não salvo: {exc}", flush=True)
                        continue
                    print(f"Revisado {source.name}: {text if text is not None else 'NEGATIVO'}", flush=True)
                    saved += 1
                    index += 1
                    break
    finally:
        cv2.destroyAllWindows()
    print(f"Sessão concluída: {saved} imagens revisadas.")
    return 0
