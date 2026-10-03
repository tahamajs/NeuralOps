Module neuralops.tests.test_file_structure
==========================================
Unit tests for framework directory integrity, python compilation, and clean workspace.

Classes
-------

`TestFileStructureAndCompilation()`
:   Verify directory structure, bytecode compilation, and absence of AppleDouble artifacts.

    ### Methods

    `framework_root(self)`
    :

    `test_all_framework_python_files_compile(self, framework_root)`
    :   Verify all python files in the framework compile to bytecode without syntax errors.

    `test_duplicate_files_are_canonical_reexports(self, framework_root)`
    :   Files present in both attentions and compression_engine must be clean re-exports.

    `test_no_appledouble_dot_underscore_files(self, framework_root)`
    :   macOS creates hidden '._*' files during copying; assert none exist in framework.

    `test_registries_healthy_counts(self)`
    :   Verify both attention and compressor registries contain the required minimal set.

    `test_required_directories_exist(self, framework_root)`
    :