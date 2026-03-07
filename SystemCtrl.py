from imports import *

def clear_pycache(folder=None):
    folder = folder or os.path.dirname(__file__)

    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return

    deleted = 0
    for root, dirs, files in os.walk(folder):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"Deleted {pycache_path}")
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {pycache_path}: {e}")

    print(f"Cleared {deleted} __pycache__ folders under {folder}")


if __name__ == "__main__":
    pass