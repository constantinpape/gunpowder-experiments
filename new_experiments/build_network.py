import sys
from backend import build_unet_dtu2, build_unet_mala


if __name__ == '__main__':
    network_key = sys.argv[1]
    if network_key == 'mala':
        print("Building mala network")
        build_unet_mala()
    elif network_key == 'dtu2':
        print("Building dtu2 network")
        build_unet_dtu2()
    else:
        raise RuntimeError("Invalid network key!")
