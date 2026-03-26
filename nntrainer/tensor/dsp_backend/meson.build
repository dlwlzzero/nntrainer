nntrainer_inc += include_directories('include')
nntrainer_inc_abs += meson.current_source_dir() / 'include'

dsp_backend_htp_sources = [] # TBD

if get_option('enable-htp')
  # Build directory for DSP code
  dsp_build_dir = meson.current_build_dir() / 'htp_dsp'

  # Custom target to build DSP library
  custom_target('htp_dsp',
    input: dsp_backend_htp_sources,
    output: 'lib_htp_dsp.a',
    command: [
      meson.current_source_dir() / 'build_dsp.sh',
      meson.current_build_dir()
    ],
    build_by_default: true
  )

endif