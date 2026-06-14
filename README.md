# Камерын дүрсээс 3D модел загварчлах систем

## Төслийн товч танилцуулга

Энэхүү төсөл нь олон өнцгөөс авсан 2D зургуудыг ашиглан автомат байдлаар 3D загвар үүсгэх систем юм. Систем нь COLMAP болон OpenMVS хэрэгслүүдийг ашиглан Structure-from-Motion (SfM) болон Multi-View Stereo (MVS) аргуудаар 3D реконструкц хийдэг.

Хэрэглэгч зураг оруулсны дараа бүх реконструкцийн үе шатуудыг график хэрэглэгчийн интерфейс (GUI)-ээр удирдан ажиллуулах боломжтой бөгөөд эцэст нь текстуртай 3D загвар үүсгэнэ.

---

## Оюутны мэдээлэл

| Талбар        | Мэдээлэл                                   |
| ------------- |--------------------------------------------|
| Сургууль      | Шинэ Монгол Технологийн Коллеж             |
| Тэнхим        | Компьютерын Ухааны Тэнхим                  |
| Сэдэв         | Камерын дүрсээс 3D модел загварчлах систем |
| Оюутан        | Б. Эрдэнэбаяр                              |
| Оюутны дугаар | s21c008b                                   |
| Он            | 2026                                       |

---

## Ашигласан технологи

* Python
* PyQt5
* COLMAP
* OpenMVS
* YAML
* OpenCV
* NumPy

---

## Системийн архитектур

Систем нь дараах үндсэн үе шатуудаас бүрдэнэ.

```text
Зураг оруулах
      ↓
Feature Extraction
      ↓
Feature Matching
      ↓
Sparse Reconstruction (COLMAP)
      ↓
Dense Reconstruction (OpenMVS)
      ↓
Mesh Generation
      ↓
Mesh Refinement
      ↓
Texture Mapping
      ↓
OBJ / PLY форматтай 3D загвар
```

---

## Үндсэн функцууд

* Олон зураг импортлох
* COLMAP ашиглан Sparse Reconstruction хийх
* OpenMVS ашиглан Dense Reconstruction хийх
* Mesh үүсгэх
* Mesh сайжруулах
* Texture Mapping хийх
* OBJ болон PLY формат экспортлох
* GUI ашиглан бүх pipeline-г удирдах

---

## Hardware Requirements

* Intel Core i5 эсвэл түүнээс дээш
* 16GB RAM ба түүнээс дээш
* NVIDIA GPU (сонголтоор)
* 20GB сул дискний зай

---

## Software Requirements

* Windows 10 / Windows 11
* Python 3.10+
* COLMAP
* OpenMVS

---

## Суулгах заавар

### Шаардлагатай програмууд

1. COLMAP суулгах
2. OpenMVS суулгах
3. Python сангууд суулгах

```bash
pip install -r requirements.txt
```

### Тохиргоо

`config/config.yaml` файлд COLMAP болон OpenMVS-ийн замыг тохируулна.

```yaml
executables:
  colmap: "C:/Program Files/COLMAP/COLMAP.bat"
  interface_colmap: "C:/OpenMVS/bin/InterfaceCOLMAP.exe"
```

---

## Ажиллуулах

```bash
python main.py
```

---

## Гаралтын файлууд

Систем амжилттай ажилласны дараа дараах файлууд үүснэ.

| Файл                  | Тайлбар                  |
| --------------------- | ------------------------ |
| dense_point_cloud.ply | Нягт цэгэн үүл           |
| mesh.ply              | 3D торон загвар          |
| textured_mesh.obj     | Текстуртай эцсийн загвар |
| textured_mesh.mtl     | Материалын мэдээлэл      |

---

## Docs

Системийн архитектурын зураг, интерфейсийн зураг, туршилтын үр дүн болон демо материалуудыг `/docs` хавтаст байршуулсан.

---

## Репозиторийн бүтэц

```text
project/
├── src/
├── config/
├── docs/
├── workspace/
├── requirements.txt
├── main.py
└── README.md
```
