# Image Detection Bypass Utility (IDBU)

一个用于图像处理和检测绕过的工具。

## 说明

本项目基于 [PurinNyova/Image-Detection-Bypass-Utility](https://github.com/PurinNyova/Image-Detection-Bypass-Utility) 改进并汉化。

这是一个**顺手项目**，代码不成熟，仅供参考和学习使用。

## 功能

- 图像后处理
- 相机模拟器
- 图像检测绕过技术

## 界面预览

![界面截图](截屏2026-05-24%2022.39.51.png)

## 编译命令

```bash
cd "/Users/huangwei/素材/Image-Detection-Bypass-Utility-idbuv14r1" && pyinstaller IDBU.spec --clean 2>&1
```

## 依赖

- Python 3.x
- PyQt5
- OpenCV
- PyTorch
- scikit-image
- numpy

## 使用方式

运行编译后的 `IDBU` 可执行文件即可启动图形界面。

## 配置

配置文件为 `config.ini`，包含图像处理的各项参数设置。

## 许可证

MIT License
