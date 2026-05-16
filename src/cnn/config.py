"""
Konfigurasi 16 variasi eksperimen CNN.
  - Jumlah layer konvolusi : 2 variasi  (2, 3)
  - Banyak filter per layer: 2 variasi  ([32,64], [64,128])
  - Ukuran filter          : 2 variasi  (3, 5)
  - Jenis pooling          : 2 variasi  (max, average)
"""


def _make_name(conv_layers, filters, kernel_sizes, pooling):
    f_str = "-".join(map(str, filters))
    k_str = "-".join(map(str, kernel_sizes))
    return f"conv{conv_layers}_filters_{f_str}_kernels_{k_str}_pool_{pooling}"


def get_all_configs():
    """
    {
        "name":         str,
        "conv_layers":  int,
        "filters":      list[int],
        "kernel_sizes": list[int],
        "pooling":      "max" | "average"
    }
    """
    configs = []

    for n_layers in [2, 3]:
        for filters_base in [[32, 64], [64, 128]]:
            for kernel_size in [3, 5]:
                for pooling in ["max", "average"]:
                    if n_layers == 2:
                        filters = list(filters_base)
                    else:
                        # Layer ke-3 kelipatan 2 dari filter terakhir
                        filters = list(filters_base) + [filters_base[-1] * 2]

                    kernel_sizes = [kernel_size] * n_layers

                    name = _make_name(n_layers, filters, kernel_sizes, pooling)

                    configs.append({
                        "name": name,
                        "conv_layers": n_layers,
                        "filters": filters,
                        "kernel_sizes": kernel_sizes,
                        "pooling": pooling,
                    })

    return configs


def get_config_by_name(name):
    for cfg in get_all_configs():
        if cfg["name"] == name:
            return cfg
    raise ValueError(f"Config '{name}' tidak ditemukan.")
