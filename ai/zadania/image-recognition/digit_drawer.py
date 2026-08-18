import sys
import tempfile
from pathlib import Path
import numpy as np

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from feed_forward import FeedForwardNetwork
from inference import load_image_to_mnist_vector


class DrawingPad(QWidget):
    def __init__(self, size=28, pen_width=2):
        super().__init__()
        self.size = size
        self.pen_width = pen_width
        self.image = QImage(self.size, self.size, QImage.Format_Grayscale8)
        self.image.fill(0)
        self.last_point = QPoint()
        self.setMinimumSize(420, 420)
        self.setStyleSheet("background-color: black; border: 1px solid #cccccc;")

    def clear(self):
        self.image.fill(0)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = self._to_image_point(event.pos())
            self._draw_at(self.last_point)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            current = self._to_image_point(event.pos())
            self._draw_line(self.last_point, current)
            self.last_point = current

    def _to_image_point(self, pos):
        x = int((pos.x() / self.width()) * self.size)
        y = int((pos.y() / self.height()) * self.size)
        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))
        return QPoint(x, y)

    def _draw_at(self, point):
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.white, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPoint(point)
        painter.end()
        self.update()

    def _draw_line(self, start, end):
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.white, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(start, end)
        painter.end()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        scaled = self.image.scaled(self.width(), self.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        painter.drawImage(0, 0, scaled)

    def save_image(self, path):
        if not path:
            return False

        mnist_image = self.image.copy()
        mnist_image.invertPixels()
        return mnist_image.save(path, "PNG")


class DigitApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digit Drawer for MNIST Inference")
        self.resize(720, 620)

        self.model_path = None
        self.drawing_pad = DrawingPad()
        self.status_label = QLabel("Nakreslite číslo 0-9 a zvoľte model pre inferenciu")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.model_label = QLabel("Model: nevybraný")
        self.model_label.setAlignment(Qt.AlignCenter)

        clear_button = QPushButton("Vymazať")
        save_button = QPushButton("Uložiť PNG")
        select_model_button = QPushButton("Vybrať model")
        predict_button = QPushButton("Inferovať")

        clear_button.clicked.connect(self.drawing_pad.clear)
        save_button.clicked.connect(self.save_image)
        select_model_button.clicked.connect(self.select_model)
        predict_button.clicked.connect(self.predict_current_digit)

        button_row = QHBoxLayout()
        button_row.addWidget(clear_button)
        button_row.addWidget(save_button)
        button_row.addWidget(select_model_button)
        button_row.addWidget(predict_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.model_label)
        main_layout.addWidget(self.drawing_pad)
        main_layout.addLayout(button_row)
        self.setLayout(main_layout)

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Vybrať model",
            "",
            "Model files (*.npy)",
        )
        if not path:
            return

        self.model_path = path
        self.model_label.setText(f"Model: {Path(path).name}")
        self.status_label.setText(f"Model bol vybraný: {path}")

    def save_image(self):
        default_name = "digit.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Uložiť obrázok",
            default_name,
            "PNG image (*.png)",
        )
        if not path:
            return

        if self.drawing_pad.save_image(path):
            self.status_label.setText(f"Uložené: {path}")
        else:
            QMessageBox.warning(self, "Chyba", "Obrázok sa nepodarilo uložiť.")

    def predict_current_digit(self):
        if not self.model_path:
            QMessageBox.warning(self, "Chyba", "Najprv vyber model (.npy).")
            return

        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name

            self.drawing_pad.save_image(temp_path)
            x = load_image_to_mnist_vector(temp_path, target_size=(28, 28), invert=True).reshape(1, -1)
            model = FeedForwardNetwork.load(self.model_path)
            probabilities = model.predict_proba(x)
            prediction = int(model.predict(x)[0])
            confidence = float(np.max(probabilities[0]))

            self.status_label.setText(
                f"Predikcia: {prediction} | istota: {confidence:.2%}"
            )
        except Exception as exc:
            QMessageBox.warning(self, "Chyba pri inferencii", str(exc))


def main():
    app = QApplication(sys.argv)
    window = DigitApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
