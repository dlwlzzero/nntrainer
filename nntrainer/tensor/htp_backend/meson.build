nntrainer_inc += include_directories('include')
nntrainer_inc_abs += meson.current_source_dir() / 'include'

hexagon_sdk_home = run_command('printenv', 'HEXAGON_SDK_HOME', check: true).stdout().strip()

# Hexagon SDK headers (remote.h for CDSP_DOMAIN_ID, etc.)
hexagon_sdk_inc_dirs = [
  hexagon_sdk_home / 'incs',
  hexagon_sdk_home / 'incs' / 'stddef',
]

nntrainer_inc += include_directories(hexagon_sdk_inc_dirs)
foreach d : hexagon_sdk_inc_dirs
  nntrainer_inc_abs += d
endforeach

# HexKL SDKL addon headers and pre-built libraries
hexkl_addon_home = hexagon_sdk_home / 'addons' / 'hexkl_addon'
hexkl_inc_dir = hexkl_addon_home / 'include'
nntrainer_inc += include_directories(hexkl_inc_dir)
nntrainer_inc_abs += hexkl_inc_dir

if get_option('enable-htp')
  # Determine the target architecture library directory
  hexkl_lib_dir = hexkl_addon_home / 'lib'
  if get_option('platform') == 'android'
    hexkl_host_lib_dir = hexkl_lib_dir / 'armv8_android26'
  else
    hexkl_host_lib_dir = hexkl_lib_dir / 'armv8_qclinux'
  endif

  # Install pre-built HexKL libraries (no custom DSP build needed)
  install_data(
    hexkl_host_lib_dir / 'libsdkl.so',
    install_dir: nntrainer_libdir
  )
endif
