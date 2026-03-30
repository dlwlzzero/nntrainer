nntrainer_inc += include_directories('include')
nntrainer_inc_abs += meson.current_source_dir() / 'include'

hexagon_sdk_home = run_command('printenv', 'HEXAGON_SDK_HOME', check: true).stdout().strip()

hexagon_sdk_inc_dirs = [
  hexagon_sdk_home / 'incs',
  hexagon_sdk_home / 'incs' / 'stddef',
  hexagon_sdk_home / 'ipc' / 'fastrpc' / 'rpcmem' / 'inc',
  hexagon_sdk_home / 'rtos' / 'qurt',
  hexagon_sdk_home / 'utils' / 'examples' 
]

nntrainer_inc += include_directories(hexagon_sdk_inc_dirs)
foreach d : hexagon_sdk_inc_dirs
  nntrainer_inc_abs += d
endforeach

if get_option('enable-htp')
  # Build directory for DSP code
  dsp_build_dir = meson.current_build_dir() / 'htp_dsp'

  # Run target to build DSP library
  custom_target('htp_dsp',
    output: 'libhtp_ops.so',
    command: [
        meson.current_source_dir() / 'build_htp.sh',
        meson.current_build_dir()
    ],
    build_by_default: true
  )
endif