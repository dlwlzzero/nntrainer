htp_common_sources = [
  'unittest_htp_common.cpp',
]

htp_test_targets = [
  'unittest_htp_mat_mul',
  'unittest_htp_rms_norm',
  'unittest_htp_quantizer',
]

foreach test_name : htp_test_targets
  exe = executable(
    test_name,
    [test_name + '.cpp'] + htp_common_sources,
    dependencies: unittest_nntrainer_deps,
    install: get_option('enable-test'),
    install_dir: application_install_dir
  )
  test(test_name, exe,
    args: ['--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), test_name)],
    timeout: test_timeout,
    suite: 'unittests'
  )
endforeach
