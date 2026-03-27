nntrainer_inc += include_directories('include')
nntrainer_inc_abs += meson.current_source_dir() / 'include'

if get_option('enable-htp')
  # Build directory for DSP code
  dsp_build_dir = meson.current_build_dir() / 'htp_dsp'

  # Run target to build DSP library
  custom_target('htp_dsp',
    output: 'libhtp_ops.so',
    command: [
        meson.current_source_dir() / 'build_dsp.sh',
        meson.current_build_dir()
    ],
    build_by_default: true
  )
endif