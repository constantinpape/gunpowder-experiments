import sys
from backend import build_unet_dtu2, build_unet_mala, build_unet_multiscale
from backend import build_unet_dtu2_inference, build_unet_mala_inference


if __name__ == '__main__':
    network_key = sys.argv[1]
    have_weight = bool(int(sys.argv[2]))
    if network_key == 'mala':
        print("Building mala network")
        if have_weight:
            print("WITH loss weighting")
        else:
            print("WITHOUT loss weighting")
        build_unet_mala(have_weights=have_weight)
    elif network_key == 'dtu2':
        print("Building dtu2 network")
        if have_weight:
            print("WITH loss weighting")
        else:
            print("WITHOUT loss weighting")
        build_unet_dtu2(have_weights=have_weight)
    elif network_key == 'multiscale':
        assert not have_weight, "Weighting currently not supported for multiscale unet"
        build_unet_multiscale(have_weights=have_weight)
    elif network_key == 'mala_inference':
        build_unet_mala_inference()
    elif network_key == 'dtu2_inference':
        build_unet_dtu2_inference()
    else:
        raise RuntimeError("Invalid network key!")
