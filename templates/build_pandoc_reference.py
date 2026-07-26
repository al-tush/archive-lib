"""
Пересобрать pandoc_reference.docx (стилевой шаблон для pandoc --reference-doc)
в этой же папке. Запуск: python3 build_pandoc_reference.py
Требует: pip install --user python-docx ; системный pandoc в PATH.
"""
import os
import subprocess
import docx
from docx.shared import Pt, RGBColor, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.style import WD_STYLE_TYPE

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(HERE, '_pandoc_default_reference.docx')
OUT_PATH = os.path.join(HERE, 'pandoc_reference.docx')

subprocess.run(['pandoc', '-o', BASE_PATH, '--print-default-data-file', 'reference.docx'], check=True)

d = docx.Document(BASE_PATH)
styles = d.styles

def get_or_create(name, style_type=WD_STYLE_TYPE.PARAGRAPH, base=None):
    try:
        s = styles[name]
    except KeyError:
        s = styles.add_style(name, style_type)
        if base is not None:
            s.base_style = base
    return s

def set_shd(style, fill):
    pPr = style.element.get_or_add_pPr()
    old = pPr.find(qn('w:shd'))
    if old is not None:
        pPr.remove(old)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill)
    pPr.append(shd)

def set_bottom_border(style, color='1F3A5F', sz=8):
    pPr = style.element.get_or_add_pPr()
    old = pPr.find(qn('w:pBdr'))
    if old is not None:
        pPr.remove(old)
    pbdr = OxmlElement('w:pBdr')
    el = OxmlElement('w:bottom')
    el.set(qn('w:val'), 'single')
    el.set(qn('w:sz'), str(sz))
    el.set(qn('w:space'), '6')
    el.set(qn('w:color'), color)
    pbdr.append(el)
    pPr.append(pbdr)

def set_box_border(style, color='D0D0D0', sz=4):
    pPr = style.element.get_or_add_pPr()
    old = pPr.find(qn('w:pBdr'))
    if old is not None:
        pPr.remove(old)
    pbdr = OxmlElement('w:pBdr')
    for edge in ('top','left','bottom','right'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), str(sz))
        el.set(qn('w:space'), '4')
        el.set(qn('w:color'), color)
        pbdr.append(el)
    pPr.append(pbdr)

NAVY = RGBColor(0x1F,0x3A,0x5F)
SLATE = RGBColor(0x33,0x33,0x33)
BODY_FONT = 'PT Sans'
MONO_FONT = 'Consolas'

normal = styles['Normal']
normal.font.name = BODY_FONT
normal.font.size = Pt(11)
normal.font.color.rgb = SLATE
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.2

bt = styles['Body Text']
bt.font.name = BODY_FONT
bt.font.size = Pt(11)
bt.paragraph_format.space_after = Pt(6)
bt.paragraph_format.line_spacing = 1.2

fp = styles['First Paragraph']
fp.font.name = BODY_FONT
fp.font.size = Pt(11)
fp.paragraph_format.space_after = Pt(6)
fp.paragraph_format.line_spacing = 1.2

compact = styles['Compact']
compact.font.name = BODY_FONT
compact.font.size = Pt(11)
compact.paragraph_format.space_after = Pt(4)
compact.paragraph_format.line_spacing = 1.15

h1 = styles['Heading 1']
h1.font.name = BODY_FONT
h1.font.size = Pt(19)
h1.font.bold = True
h1.font.color.rgb = NAVY
h1.paragraph_format.space_before = Pt(4)
h1.paragraph_format.space_after = Pt(10)
set_bottom_border(h1)

h2 = styles['Heading 2']
h2.font.name = BODY_FONT
h2.font.size = Pt(14.5)
h2.font.bold = True
h2.font.color.rgb = NAVY
h2.paragraph_format.space_before = Pt(14)
h2.paragraph_format.space_after = Pt(6)

h3 = styles['Heading 3']
h3.font.name = BODY_FONT
h3.font.size = Pt(12.5)
h3.font.bold = True
h3.font.color.rgb = SLATE
h3.paragraph_format.space_before = Pt(10)
h3.paragraph_format.space_after = Pt(4)

title = styles['Title']
title.font.name = BODY_FONT
title.font.size = Pt(24)
title.font.bold = True
title.font.color.rgb = NAVY

sc = get_or_create('Source Code', WD_STYLE_TYPE.PARAGRAPH, base=normal)
sc.font.name = MONO_FONT
sc.font.size = Pt(9.5)
sc.font.color.rgb = RGBColor(0x1A,0x1A,0x1A)
sc.paragraph_format.space_before = Pt(6)
sc.paragraph_format.space_after = Pt(6)
sc.paragraph_format.left_indent = Cm(0.4)
sc.paragraph_format.line_spacing = 1.05
set_shd(sc, 'F2F2F2')
set_box_border(sc)

vc = styles['Verbatim Char']
vc.font.name = MONO_FONT
vc.font.size = Pt(10.5)
vc.font.color.rgb = RGBColor(0x8A,0x2B,0x06)

d.save(OUT_PATH)
os.remove(BASE_PATH)
print('written', OUT_PATH)
