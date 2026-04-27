import sys
import os

# Ensure src is in path if needed (though syn3d_llm package structure should handle it)
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.ui.app import Syn3dApp

def main():
    app = Syn3dApp()
    app.mainloop()

if __name__ == "__main__":
    main()
