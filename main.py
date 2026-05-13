import os
import faulthandler

# StackOverflow'daki tavsiye: C++ bellek ihlallerini (0xC0000005) terminale yazdırır
faulthandler.enable()

# Zorunlu ortam değişkenleri (KESİNLİKLE importlardan önce tanımlanmalı)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "gl"

import sys
import traceback
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.gui import ODin3DPC_GUI

def main():
    try:
        # Qt'nin sistem GPU sürücüsünü kullanmasını zorla
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)

        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = ODin3DPC_GUI()
        window.show()
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()