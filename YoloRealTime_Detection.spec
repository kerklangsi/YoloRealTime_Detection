# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['YoloRealTime_Detection.py'],
    pathex=[],
    binaries=[],
    datas=[('detection', 'detection'), ('gui', 'gui'), ('icons', 'icons'), ('models', 'models')],
    hiddenimports=['ultralytics', 'torch', 'opencv-python', 'numpy', 'Pillow', 'pygrabber'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YoloRealTime_Detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['E:\\Github\\YoloRealTime_Detection\\icons\\YoloRealTime_Detection.ico'],
    contents_directory='dist\YoloRealTime_Detection',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YoloRealTime_Detection',
)
