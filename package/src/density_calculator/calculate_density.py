from .difference_density_module import run, Input, Output
import argparse

def main():
    parser = argparse.ArgumentParser(description='nucleofind build')
    parser.add_argument("-mtzin", required=True)
    parser.add_argument("-pdbin", required=True)

    args = parser.parse_args()  

    i = Input(args.mtzin, args.pdbin , "FP,SIGFP", "")
    o = Output("s")
    run(i, o)
    # run(*aparser.parse_args())

if __name__ == "__main__":
    main()