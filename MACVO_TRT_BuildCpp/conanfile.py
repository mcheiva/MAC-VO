from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout


class MacvoTrtBuilderRecipe(ConanFile):
    name = "macvo_trt_builder"
    version = "0.1.0"
    package_type = "application"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"
    exports_sources = "CMakeLists.txt", "include/*", "src/*"

    def requirements(self):
        self.requires("cli11/2.4.2")

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
