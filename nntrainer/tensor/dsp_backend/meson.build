nntrainer_inc += include_directories('include')
nntrainer_inc_abs += meson.current_source_dir() / 'include'

if get_option('enable-htp')
  # Build directory for DSP code
  dsp_build_dir = meson.current_build_dir() / 'htp_dsp'

  # Run target to build DSP library
  run_target(
    'htp_dsp',
    command: [
      'sh',
      meson.current_source_dir() / 'build_dsp.sh',
      meson.current_build_dir()
    ]
  )

endif