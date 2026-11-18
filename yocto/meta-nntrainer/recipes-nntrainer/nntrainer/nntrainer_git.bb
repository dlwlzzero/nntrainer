SUMMARY = "NNTrainer: on-device neural network training and inference"
HOMEPAGE = "https://github.com/nntrainer/nntrainer"
SECTION = "libs"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://LICENSE;md5=095e13fef457e259d3bc155d0ed859f1 \
                    file://debian/copyright;md5=06434e412e1de1e75fbd4f42920ebc4e"

# Used when building outside CI. CI replaces the source with the checked-out
# tree through externalsrc (EXTERNALSRC:pn-nntrainer).
SRC_URI = "gitsm://github.com/nntrainer/nntrainer.git;branch=main;protocol=https"
SRCREV = "${AUTOREV}"
PV = "0.6.0+git"
S = "${WORKDIR}/git"

# iniparser and ruy come from the bundled subprojects/ (meson cmake.subproject),
# so cmake-native is needed and no iniparser recipe is required.
DEPENDS = "openblas ml-api cmake-native"

inherit meson pkgconfig

EXTRA_OEMESON = " \
    -Dplatform=yocto \
    -Denable-blas=true \
    -Denable-app=false \
    -Dinstall-app=false \
    -Denable-capi=enabled \
    -Denable-ccapi=true \
    -Dml-api-support=enabled \
    -Denable-test=false \
    -Denable-logging=true \
    -Denable-nnstreamer-tensor-filter=disabled \
    -Denable-nnstreamer-tensor-trainer=disabled \
    -Denable-nnstreamer-backbone=false \
    -Denable-tflite-backbone=false \
    -Denable-tflite-interpreter=false \
"

# nntrainer installs unversioned shared libraries (libnntrainer.so,
# libccapi-nntrainer.so, libcapi-nntrainer.so); ship them in the main package.
FILES_SOLIBSDEV = ""
FILES:${PN} += "${libdir}/*.so"
INSANE_SKIP:${PN} += "dev-so"
