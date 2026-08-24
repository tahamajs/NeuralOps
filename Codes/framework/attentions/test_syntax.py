import py_compile
import glob
for f in glob.glob('/Users/tahamajs/Documents/uni/Project/Codes/framework/attentions/aegis_*.py'):
    py_compile.compile(f)
