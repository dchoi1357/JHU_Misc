import sys

def main():
    if len(sys.argv) > 1:
        print("Args: %s"%sys.argv)
    else:
        print("no args")
        
if __name__ == "__main__":
    main()