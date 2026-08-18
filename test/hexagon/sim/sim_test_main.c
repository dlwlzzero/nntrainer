// SPDX-License-Identifier: Apache-2.0
/**
 * @file	sim_test_main.c
 * @date	18 August 2026
 * @brief	Entry point dispatching hexagon-sim tests by name (argv[1])
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdio.h>
#include <string.h>
int test_smoke(void);
int test_pool(void);
/* Each task adds one extern declaration here and one table entry below. */
static const struct {
  const char *name;
  int (*fn)(void);
} tests[] = {
  {"smoke", test_smoke},
  {"pool", test_pool},
};
int main(int argc, char **argv) {
  if (argc < 2) {
    printf("SIM_TEST usage: <name>\n");
    return 2;
  }
  for (unsigned i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i)
    if (!strcmp(argv[1], tests[i].name))
      return tests[i].fn();
  printf("SIM_TEST unknown test: %s\n", argv[1]);
  return 2;
}
