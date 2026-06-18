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
  # Build HTP libraries via cmake (for target device deployment)
  # Host-side libhtp_ops.so and DSP-side libhtp_ops_skel.so
  htp_lib = custom_target('htp_lib',
    output: ['libhtp_ops.so', 'libhtp_ops_skel.so'],
    command: [
        meson.current_source_dir() / 'build_htp.sh',
        meson.current_build_dir()
    ],
    build_by_default: true,
    install: true,
    install_dir: [nntrainer_libdir, nntrainer_libdir]
  )
endif
