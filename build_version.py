"""build_version.py - writes version_info.txt for PyInstaller, so SubZeroAstro.exe carries
its product name, version and author in its Windows file properties (also checked by
SignPath's signing policy). Run by the GitHub build:  python build_version.py"""
import re
from pathlib import Path

src = Path(__file__).with_name("subzeroastro.py").read_text(encoding="utf-8")
ver = re.search(r'^APP_VERSION = "([^"]+)"', src, re.M).group(1)
author = re.search(r'^APP_AUTHOR = "([^"]+)"', src, re.M).group(1)
nums = [int(x) for x in re.findall(r"\d+", ver)[:3]] + [0, 0, 0]
v4 = tuple(nums[:3] + [0])
Path("version_info.txt").write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={v4}, prodvers={v4}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', '{author}'),
      StringStruct('FileDescription', 'SubZeroAstro - from subs to masters, zero hassle'),
      StringStruct('FileVersion', '{ver}'),
      StringStruct('InternalName', 'SubZeroAstro'),
      StringStruct('LegalCopyright', 'Copyright (c) 2026 {author}. MIT License.'),
      StringStruct('OriginalFilename', 'SubZeroAstro.exe'),
      StringStruct('ProductName', 'SubZeroAstro'),
      StringStruct('ProductVersion', '{ver}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""", encoding="utf-8")
print(f"version_info.txt for SubZeroAstro {ver} ({'.'.join(map(str, v4))})")
