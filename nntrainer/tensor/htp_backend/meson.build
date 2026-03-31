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
  # Build host-side HTP library directly with meson
  # cdsprpc (FastRPC) is resolved at runtime on target device
  htp_host_lib = shared_library('htp_ops',
    [
      meson.current_source_dir() / 'include' / 'host' / 'htp_ops_stub.c',
      hexagon_sdk_home / 'utils' / 'examples' / 'dsp_capabilities_utils.c',
      meson.current_source_dir() / 'host' / 'op_export.c',
      meson.current_source_dir() / 'host' / 'session.c',
    ],
    include_directories: nntrainer_inc,
    link_args: ['-lcdsprpc'],
    install: true,
    install_dir: get_option('libdir')
  )

  htp_dep = declare_dependency(link_with: htp_host_lib)

  # Build DSP-side library (for device deployment, not linked into nntrainer)
  custom_target('htp_dsp_lib',
    output: 'libhtp_ops_skel.so',
    command: [
        meson.current_source_dir() / 'build_htp.sh',
        meson.current_build_dir()
    ],
    build_by_default: true
  )
endif
