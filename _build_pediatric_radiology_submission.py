from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image


ROOT = Path(r"E:\Softwares\PycharmProject\Radiomics")
SOURCE_DIR = ROOT / "European Journal of Pediatrics"
SOURCE_DOCX = SOURCE_DIR / "Manuscript_EJPwithAPP.docx"
OUT = ROOT / "Pediatric Radiology"
FIGURES = OUT / "Figures"
SUPPLEMENTARY = OUT / "Supplementary"

TITLE = (
    "Prediction of refractory Mycoplasma pneumoniae pneumonia in children using "
    "chest radiography-based radiomics and machine learning"
)
RUNNING_TITLE = "RMPP prediction using chest radiography"

AUTHORS = [
    "Bofan Wang",
    "Maoxiang Zhang",
    "Rufeng Ji",
    "Deyu Zhao",
    "Xia Huang",
]
AFFILIATIONS = [
    "Children's Hospital of Nanjing Medical University, Nanjing 211100, Jiangsu, China",
    (
        "Department of Respiratory Medicine, Children's Hospital of Nanjing Medical "
        "University, Nanjing, Jiangsu, China; Department of Pediatrics, Nanjing Lishui "
        "People's Hospital, Zhongda Hospital Lishui Branch, Southeast University, "
        "Nanjing, China"
    ),
]
CORRESPONDENCE = (
    "Deyu Zhao, zhaodeyu988@126.com; Xia Huang, huang_xia01@163.com"
)

ABSTRACT = [
    (
        "Background",
        (
            "Refractory Mycoplasma pneumoniae pneumonia (RMPP) is a severe form of "
            "community-acquired pneumonia in children. Early identification is "
            "important for improving prognosis."
        ),
    ),
    (
        "Objective",
        (
            "To develop and evaluate chest radiography-based radiomics models combined "
            "with machine learning for early identification of RMPP in children."
        ),
    ),
    (
        "Materials and methods",
        (
            "This retrospective, single-center study included chest radiographs from "
            "children with Mycoplasma pneumoniae pneumonia (MPP). Radiomics features "
            "were extracted with PyRadiomics, and feature selection was performed "
            "within each cross-validation fold. Nine machine learning models were "
            "constructed and compared using discrimination, calibration, decision "
            "curve analysis, and Shapley additive explanations. A web-based interface "
            "was developed to provide individualized risk estimates from DICOM images "
            "and routine clinical variables."
        ),
    ),
    (
        "Results",
        (
            "A total of 931 children with MPP were included (557 with RMPP and 374 "
            "without RMPP). Of 102 extracted radiomics features, 98 (96.1%) showed "
            "good-to-excellent reproducibility (intraclass correlation coefficient > "
            "0.75; mean 0.932). Within each fold, minimum redundancy maximum relevance "
            "selection retained nine radiomics features, which were combined with age, "
            "C-reactive protein, white blood cell count, and lactate dehydrogenase. "
            "Among the nine models, LightGBM was the most robust, with a validation "
            "area under the receiver operating characteristic curve of 0.757 (95% CI "
            "0.691-0.825), recall of 0.857, precision of 0.706, and accuracy of 0.701. "
            "Decision curve analysis confirmed clinical net benefit, and Shapley "
            "additive explanations identified the main contributing features."
        ),
    ),
    (
        "Conclusion",
        (
            "A LightGBM model based on chest radiography radiomics showed moderate "
            "discriminative ability and clinical utility for predicting RMPP in "
            "children and was packaged into a web-based interface for bedside use."
        ),
    ),
]

KEYWORDS = (
    "Mycoplasma pneumoniae pneumonia; children; radiomics; machine learning; "
    "chest radiography; prediction model"
)


def normalize_text(text: str) -> str:
    replacements = {
        "Supplementary Fig.1 and Supplementary Table 1": "Online Resources 2 and 3",
        "Supplementary Fig. 1 and Supplementary Table 1": "Online Resources 2 and 3",
        "Supplementary Table 2": "Online Resource 4",
        "Supplementary Table 3": "Online Resource 5",
        "Supplementary Table 4": "Online Resource 6",
        "Supplementary File": "Online Resource 1",
        "Supplementary Fig. 1": "Online Resource 2",
        "Supplementary Table 1": "Online Resource 3",
        "48 hours": "48 h",
        "T-Tests": "t tests",
        "T-Test": "t test",
        "brier scores": "Brier scores",
        "(e)Calibration": "(e) Calibration",
        "Fig. 4)": "Fig. 4).",
        "refractory Mycoplasma pneumoniae pneumonia.": (
            "refractory Mycoplasma pneumoniae pneumonia (RMPP)."
        ),
        "Chest X-ray is the preferred imaging modality": (
            "Chest X-ray (CXR) is the preferred imaging modality"
        ),
        "Children's Hospital of Nanjing Medical University": (
            "the participating tertiary pediatric hospital"
        ),
        (
            "The study was approved by the Ethics Committee of the participating "
            "tertiary pediatric hospital (approval no. 202601046-1)."
        ): (
            "The study was approved by the institutional research ethics committee at "
            "the participating center. The approval number has been omitted from the "
            "anonymized manuscript and will be provided to the editorial office."
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def source_texts(src: Document, start: int, end: int) -> list[str]:
    return [normalize_text(p.text) for p in src.paragraphs[start:end] if p.text.strip()]


def set_run_font(run, size: float = 10, bold: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")


def set_paragraph_format(
    paragraph,
    *,
    double_spaced: bool = True,
    keep_with_next: bool = False,
    space_before: float = 0,
    space_after: float = 0,
) -> None:
    fmt = paragraph.paragraph_format
    fmt.line_spacing_rule = (
        WD_LINE_SPACING.DOUBLE if double_spaced else WD_LINE_SPACING.SINGLE
    )
    fmt.space_before = Pt(space_before)
    fmt.space_after = Pt(space_after)
    fmt.keep_with_next = keep_with_next
    fmt.widow_control = True


def configure_document(doc: Document, *, line_numbers: bool = True) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    if line_numbers:
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:restart"), "continuous")
        section._sectPr.append(ln)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    normal.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)

    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_format(p, double_spaced=False)
    run = p.add_run()
    set_run_font(run, 9)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, end])


def add_paragraph(
    doc: Document,
    text: str = "",
    *,
    bold: bool = False,
    size: float = 10,
    align=None,
    double_spaced: bool = True,
    keep_with_next: bool = False,
    italic: bool = False,
    space_before: float = 0,
    space_after: float = 0,
):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    set_paragraph_format(
        p,
        double_spaced=double_spaced,
        keep_with_next=keep_with_next,
        space_before=space_before,
        space_after=space_after,
    )
    if text:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        set_run_font(run, size, bold)
    return p


def add_heading(doc: Document, text: str, level: int = 1):
    if level == 1:
        return add_paragraph(
            doc,
            text,
            bold=True,
            size=11,
            keep_with_next=True,
            space_before=8,
        )
    if level == 2:
        return add_paragraph(
            doc,
            text,
            bold=True,
            size=10,
            keep_with_next=True,
            space_before=6,
        )
    return add_paragraph(
        doc,
        text,
        bold=False,
        italic=True,
        size=10,
        keep_with_next=True,
        space_before=4,
    )


def add_page_break(doc: Document) -> None:
    p = doc.add_paragraph()
    set_paragraph_format(p, double_spaced=False)
    p.add_run().add_break(WD_BREAK.PAGE)


def add_labeled_paragraph(doc: Document, label: str, text: str) -> None:
    p = doc.add_paragraph()
    set_paragraph_format(p)
    r = p.add_run(f"{label}: ")
    r.bold = True
    set_run_font(r, 10, True)
    r2 = p.add_run(text)
    set_run_font(r2, 10)


def set_cell_margins(cell, top=80, start=100, bottom=80, end=100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for key, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{key}"))
        if node is None:
            node = OxmlElement(f"w:{key}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:color"), "auto")


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "D9D9D9")


def add_cell_text(
    cell,
    text: str,
    *,
    bold: bool = False,
    italic_p: bool = False,
    size: float = 8.5,
    align=WD_ALIGN_PARAGRAPH.LEFT,
    superscript_marker: bool = False,
) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_cell_margins(cell)
    p = cell.paragraphs[0]
    p.alignment = align
    set_paragraph_format(p, double_spaced=False)
    has_significance_marker = text.endswith("*")
    if has_significance_marker:
        text = text[:-1]
    marker_match = None
    if superscript_marker:
        marker_match = re.match(r"^(.*?)([1-5])(\s*\(.*\))$", text)
        if marker_match is None:
            marker_match = re.match(r"^(.*?)([1-5])$", text)
    if marker_match:
        base = marker_match.group(1)
        if marker_match.lastindex and marker_match.lastindex >= 3:
            base += marker_match.group(3) or ""
    else:
        base = text
    marker = marker_match.group(2) if marker_match else None
    if italic_p and "p" in base:
        prefix, suffix = base.split("p", 1)
        r1 = p.add_run(prefix)
        set_run_font(r1, size, bold)
        r2 = p.add_run("p")
        r2.italic = True
        set_run_font(r2, size, bold)
        r3 = p.add_run(suffix)
        set_run_font(r3, size, bold)
    else:
        run = p.add_run(base)
        set_run_font(run, size, bold)
    if marker:
        marker_run = p.add_run(marker)
        marker_run.font.superscript = True
        set_run_font(marker_run, size)
    if has_significance_marker:
        marker_run = p.add_run("*")
        marker_run.font.superscript = True
        set_run_font(marker_run, size)


def add_table(
    doc: Document,
    caption: str,
    header: list[str],
    rows: list[list[str]],
    note: str,
    *,
    first_col_superscript: bool = False,
    narrow_font: float = 8.5,
):
    add_paragraph(doc, caption, bold=True, keep_with_next=True, space_after=6)
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    for idx, value in enumerate(header):
        cell = table.rows[0].cells[idx]
        set_cell_shading(cell, "D9E2F3")
        add_cell_text(
            cell,
            value,
            bold=True,
            size=narrow_font + 0.5,
            align=WD_ALIGN_PARAGRAPH.CENTER,
            italic_p=value.strip() == "p-value*",
        )
    set_repeat_table_header(table.rows[0])

    for row_values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row_values):
            align = (
                WD_ALIGN_PARAGRAPH.LEFT
                if idx == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            add_cell_text(
                cells[idx],
                value,
                size=narrow_font,
                align=align,
                superscript_marker=first_col_superscript and idx == 0,
            )
    add_paragraph(
        doc,
        note,
        size=8.5,
        double_spaced=False,
        space_before=6,
        space_after=6,
    )


def add_figure(doc: Document, caption: str, image_path: Path) -> None:
    p = doc.add_paragraph()
    set_paragraph_format(p, double_spaced=False, keep_with_next=True, space_before=6)
    label, rest = caption.split(" ", 1)
    label_run = p.add_run(label)
    label_run.bold = True
    set_run_font(label_run, 10, True)
    rest_run = p.add_run(" " + rest)
    set_run_font(rest_run, 10)

    image = Image.open(image_path)
    aspect = image.width / image.height
    max_w = 15.5
    max_h = 18.5
    width = min(max_w, max_h * aspect)
    pic_p = doc.add_paragraph()
    pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_format(
        pic_p,
        double_spaced=False,
        keep_with_next=False,
        space_before=4,
        space_after=12,
    )
    pic_p.add_run().add_picture(str(image_path), width=Cm(width))


def strip_final_period(text: str) -> str:
    return text.rstrip().rstrip(".")


def build_manuscript(src: Document, md: list[str]) -> tuple[int, int]:
    doc = Document()
    configure_document(doc, line_numbers=True)

    add_paragraph(
        doc,
        TITLE,
        bold=True,
        size=14,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=12,
    )
    add_heading(doc, "Abstract", 1)
    for label, text in ABSTRACT:
        add_labeled_paragraph(doc, label, text)

    add_paragraph(
        doc,
        f"Keywords: {KEYWORDS}",
        space_before=6,
    )

    add_heading(doc, "Abbreviations", 1)
    abbreviations = source_texts(src, 16, 41)
    for item in abbreviations:
        if ", " not in item:
            continue
        abbreviation, definition = item.split(", ", 1)
        p = doc.add_paragraph(style="List Bullet")
        set_paragraph_format(p)
        run = p.add_run(abbreviation)
        run.bold = True
        set_run_font(run, 10, True)
        definition_run = p.add_run(f", {definition}")
        set_run_font(definition_run, 10)

    add_heading(doc, "Introduction", 1)
    intro = source_texts(src, 43, 47)
    for paragraph in intro:
        add_paragraph(doc, paragraph)

    add_heading(doc, "Materials and methods", 1)
    methods = [
        ("h2", "Study population and data source"),
        ("p", source_texts(src, 49, 50)[0]),
        (
            "p",
            (
                "Inclusion criteria were: (a) positive etiological evidence of MPP "
                "(positive Mycoplasma pneumoniae IgM or polymerase chain reaction); "
                "(b) clinical manifestations consistent with MPP; (c) imaging findings "
                "consistent with MPP; and (d) age younger than 18 years at admission."
            ),
        ),
        ("p", source_texts(src, 51, 52)[0]),
        ("p", source_texts(src, 52, 53)[0]),
        ("h2", "Image acquisition and preprocessing"),
        ("p", source_texts(src, 54, 55)[0]),
        ("h2", "Image segmentation and reproducibility assessment"),
        ("p", source_texts(src, 56, 57)[0]),
        ("h2", "Radiomics feature extraction and reproducibility"),
        ("p", source_texts(src, 58, 59)[0]),
        ("h2", "Model construction"),
        ("h3", "Dataset partitioning"),
        ("p", source_texts(src, 61, 62)[0]),
        ("h3", "Feature selection"),
        ("p", source_texts(src, 63, 64)[0]),
        ("h3", "Machine learning model development and optimization"),
        ("p", source_texts(src, 65, 66)[0]),
        ("h3", "Comprehensive model evaluation"),
        ("p", source_texts(src, 67, 68)[0]),
        ("h2", "Web-based clinical prediction interface"),
        ("p", source_texts(src, 69, 70)[0]),
        ("h2", "Statistical analysis"),
        ("p", source_texts(src, 71, 72)[0]),
    ]
    for kind, text in methods:
        add_heading(doc, text, int(kind[1])) if kind.startswith("h") else add_paragraph(
            doc, text
        )

    add_heading(doc, "Results", 1)
    results = [
        ("h2", "Baseline characteristics"),
        ("p", source_texts(src, 74, 75)[0]),
        ("h2", "Model performance"),
        ("p", source_texts(src, 76, 77)[0]),
        ("p", source_texts(src, 77, 78)[0]),
        ("p", source_texts(src, 78, 79)[0]),
        ("p", source_texts(src, 79, 80)[0]),
        ("p", source_texts(src, 80, 81)[0]),
        ("p", source_texts(src, 81, 82)[0]),
        ("h2", "Prediction interface"),
        ("p", source_texts(src, 83, 84)[0]),
        ("h2", "SHAP and decision curve analysis"),
        ("p", source_texts(src, 85, 86)[0]),
        ("p", source_texts(src, 86, 87)[0]),
    ]
    for kind, text in results:
        add_heading(doc, text, int(kind[1])) if kind.startswith("h") else add_paragraph(
            doc, text
        )

    add_heading(doc, "Discussion", 1)
    discussion = source_texts(src, 88, 96)
    for paragraph in discussion:
        add_paragraph(doc, paragraph)

    add_heading(doc, "Conclusion", 1)
    conclusion = (
        "A chest radiography-based radiomics model combined with machine learning, "
        "together with a web-based prediction interface, may support early risk "
        "stratification for refractory Mycoplasma pneumoniae pneumonia in children. "
        "Prospective multicenter validation is needed before clinical implementation."
    )
    add_paragraph(doc, conclusion)

    add_heading(doc, "Statements and Declarations", 1)
    statements = [
        ("h2", "Ethics approval"),
        (
            "p",
            (
                "This retrospective study was approved by the institutional research "
                "ethics committee at the participating center and was performed in "
                "accordance with the ethical standards of the 1964 Declaration of "
                "Helsinki and its later amendments. The approval number has been "
                "omitted from the anonymized manuscript and will be provided to the "
                "editorial office."
            ),
        ),
        ("h2", "Consent to participate"),
        ("p", "Not applicable because of the retrospective design and use of de-identified data."),
        ("h2", "Consent for publication"),
        ("p", "Not applicable."),
        ("h2", "Data availability"),
        (
            "p",
            (
                "The image dataset analyzed during the current study is not publicly "
                "available because of participant privacy, but it is available from the "
                "corresponding author on reasonable request. Other datasets generated "
                "or analyzed during this study are included in this article and its "
                "Online Resources."
            ),
        ),
        ("h2", "Code availability"),
        (
            "p",
            (
                "The analysis code and the web-based prediction interface are available "
                "from the corresponding author on reasonable request."
            ),
        ),
    ]
    for kind, text in statements:
        add_heading(doc, text, 2) if kind == "h2" else add_paragraph(doc, text)

    add_heading(doc, "References", 1)
    references = source_texts(src, 115, 130)
    for reference in references:
        p = add_paragraph(doc, reference)
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.75)

    add_page_break(doc)
    add_heading(doc, "Tables", 1)
    table1 = src.tables[0]
    table1_header = [cell.text.strip() for cell in table1.rows[0].cells]
    table1_rows = [
        [cell.text.strip() for cell in row.cells] for row in table1.rows[1:]
    ]
    table1_caption = (
        "Table 1 Baseline and clinical characteristics of children with refractory "
        "and non-refractory Mycoplasma pneumoniae pneumonia"
    )
    table1_note = (
        "Note: Normally distributed continuous variables are reported as mean ± SD; "
        "non-normally distributed continuous variables as median (IQR); and "
        "categorical variables as number (percentage). a Welch t test; b chi-square "
        "test; c t test; d Fisher exact test; e Mann-Whitney U test. *p < 0.05 "
        "indicates statistical significance. ALT, alanine aminotransferase; APTT, "
        "activated partial thromboplastin time; AST, aspartate aminotransferase; BO, "
        "bronchiolitis obliterans; CK, creatine kinase; CK-MB, creatine kinase-MB "
        "isoenzyme; CRP, C-reactive protein; Hb, hemoglobin; LDH, lactate "
        "dehydrogenase; LYMPH, lymphocyte count; NEUT, neutrophil count; PLT, platelet "
        "count; PT, prothrombin time; RMPP, refractory Mycoplasma pneumoniae "
        "pneumonia; WBC, white blood cell count."
    )
    add_table(
        doc,
        table1_caption,
        table1_header,
        table1_rows,
        table1_note,
        first_col_superscript=True,
        narrow_font=8.0,
    )

    add_page_break(doc)
    table2 = src.tables[1]
    table2_header = [cell.text.strip() for cell in table2.rows[0].cells]
    table2_rows = [
        [cell.text.strip() for cell in row.cells] for row in table2.rows[1:]
    ]
    table2_caption = (
        "Table 2 Performance of the nine machine learning models in the training and "
        "validation sets"
    )
    table2_note = (
        "Note: AUC, area under the receiver operating characteristic curve. p values "
        "are from the DeLong test comparing each model's validation AUC with LightGBM "
        "and were corrected using the Benjamini-Hochberg false discovery rate "
        "procedure across all pairwise comparisons. LightGBM had the highest "
        "validation AUC (0.757) and recall (0.857)."
    )
    add_table(
        doc,
        table2_caption,
        table2_header,
        table2_rows,
        table2_note,
        narrow_font=7.5,
    )

    add_page_break(doc)
    add_heading(doc, "Figure legends and figures", 1)
    figure_sources = [
        SOURCE_DIR / "Figures" / "Fig1_Study_diagram.png",
        SOURCE_DIR / "Figures" / "Fig2_Study_workflow.png",
        SOURCE_DIR / "Figures" / "Fig3_Feature_extraction_modeling.png",
        SOURCE_DIR / "Supplementary" / "Supplementary_Fig4_App_workflow.png",
        SOURCE_DIR / "Figures" / "Fig4_ROC_SHAP_confusion_matrix.png",
        SOURCE_DIR / "Figures" / "Fig5_Calibration_DCA.png",
        SOURCE_DIR / "Figures" / "Fig6_DeLong_pvalues.png",
    ]
    figure_captions = [
        strip_final_period(text) for text in source_texts(src, 136, 143)
    ]
    for idx, (figure_source, caption) in enumerate(
        zip(figure_sources, figure_captions), start=1
    ):
        figure_target = FIGURES / f"Fig{idx}.png"
        shutil.copy2(figure_source, figure_target)
        if idx > 1:
            add_page_break(doc)
        add_figure(doc, caption, figure_target)

    add_page_break(doc)
    add_heading(doc, "Online Resources", 1)
    online_resources = [
        "Online Resource 1 PyRadiomics configuration (TXT)",
        (
            "Online Resource 2 Distribution of the Dice similarity coefficient for "
            "inter-rater segmentation agreement (PNG)"
        ),
        (
            "Online Resource 3 Dice similarity coefficient results for the 50-case "
            "reproducibility subset (XLSX)"
        ),
        (
            "Online Resource 4 Intraclass correlation coefficient results for all "
            "extracted radiomics features (XLSX)"
        ),
        "Online Resource 5 Detailed model performance metrics (XLSX)",
        (
            "Online Resource 6 Full pairwise DeLong test p values after false "
            "discovery rate correction (XLSX)"
        ),
    ]
    for item in online_resources:
        add_paragraph(doc, item)

    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "Anonymous manuscript for Pediatric Radiology"
    doc.core_properties.comments = ""
    output = OUT / "Manuscript_PR_anonymized.docx"
    doc.save(output)

    md.extend(
        [
            f"# {TITLE}",
            "",
            "## Abstract",
            "",
        ]
    )
    for label, text in ABSTRACT:
        md.extend([f"**{label}:** {text}", ""])
    md.extend([f"**Keywords:** {KEYWORDS}", "", "## Abbreviations", ""])
    for item in abbreviations:
        md.append(f"- {item}")
    md.extend(["", "## Introduction", ""])
    for paragraph in intro:
        md.extend([paragraph, ""])
    md.extend(["## Materials and methods", ""])
    for kind, text in methods:
        md.append(f"### {text}" if kind == "h2" else f"#### {text}" if kind == "h3" else text)
        md.append("")
    md.extend(["## Results", ""])
    for kind, text in results:
        md.append(f"### {text}" if kind == "h2" else text)
        md.append("")
    md.extend(["## Discussion", ""])
    for paragraph in discussion:
        md.extend([paragraph, ""])
    md.extend(["## Conclusion", "", conclusion, ""])
    md.extend(
        [
            "## Statements and Declarations",
            "",
            "**Ethics approval**",
            "",
            statements[1][1],
            "",
            "**Consent to participate**",
            "",
            statements[3][1],
            "",
            "**Consent for publication**",
            "",
            statements[5][1],
            "",
            "**Data availability**",
            "",
            statements[7][1],
            "",
            "**Code availability**",
            "",
            statements[9][1],
            "",
            "## References",
            "",
        ]
    )
    for reference in references:
        md.append(reference)
    md.extend(["", "## Tables", "", f"**{table1_caption}**", "", table1_note, ""])
    md.extend(
        [
            "| " + " | ".join(table1_header) + " |",
            "|" + "---|" * len(table1_header),
        ]
    )
    for row in table1_rows:
        md.append("| " + " | ".join(row) + " |")
    md.extend(["", f"**{table2_caption}**", "", table2_note, ""])
    md.extend(
        [
            "| " + " | ".join(table2_header) + " |",
            "|" + "---|" * len(table2_header),
        ]
    )
    for row in table2_rows:
        md.append("| " + " | ".join(row) + " |")
    md.extend(["", "## Figure legends", ""])
    for caption in figure_captions:
        md.extend([caption, ""])
    md.extend(["## Online Resources", ""])
    for item in online_resources:
        md.append(f"- {item}")

    body_parts = intro + [
        text for kind, text in methods if kind == "p"
    ] + [text for kind, text in results if kind == "p"] + discussion + [conclusion]
    body_word_count = sum(len(re.findall(r"\S+", text)) for text in body_parts)
    abstract_word_count = sum(
        len(re.findall(r"\S+", text)) for _, text in ABSTRACT
    )
    return abstract_word_count, body_word_count


def build_title_page(abstract_words: int, body_words: int) -> None:
    doc = Document()
    configure_document(doc, line_numbers=False)

    add_paragraph(
        doc,
        "Title Page",
        bold=True,
        size=12,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=8,
    )
    add_paragraph(
        doc,
        TITLE,
        bold=True,
        size=14,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=12,
    )
    add_paragraph(doc, "Article type: Research", bold=True, space_after=6)
    add_paragraph(doc, "Authors", bold=True, keep_with_next=True)
    for author, affiliation, orcid in zip(
        AUTHORS,
        ["1", "1", "2", "1", "1"],
        ["[to be provided]"] * len(AUTHORS),
    ):
        p = doc.add_paragraph()
        set_paragraph_format(p)
        author_run = p.add_run(author)
        set_run_font(author_run, 10)
        affiliation_run = p.add_run(affiliation)
        affiliation_run.font.superscript = True
        set_run_font(affiliation_run, 10)
        if author in {"Deyu Zhao", "Xia Huang"}:
            corresponding_run = p.add_run("*")
            corresponding_run.font.superscript = True
            set_run_font(corresponding_run, 10)
        orcid_run = p.add_run(f" (ORCID iD: {orcid})")
        set_run_font(orcid_run, 10)

    add_paragraph(doc, "Affiliations", bold=True, keep_with_next=True, space_before=6)
    for idx, affiliation in enumerate(AFFILIATIONS, start=1):
        add_paragraph(doc, f"{idx} {affiliation}")

    add_paragraph(
        doc,
        "Corresponding author",
        bold=True,
        keep_with_next=True,
        space_before=6,
    )
    add_paragraph(doc, CORRESPONDENCE)
    add_paragraph(doc, "*Corresponding authors")
    add_paragraph(doc, f"Running title: {RUNNING_TITLE}")
    add_paragraph(
        doc,
        f"Abstract: {abstract_words} words; Introduction-Conclusion: {body_words} words; "
        "7 figures; 2 tables; 6 Online Resources.",
    )

    add_heading(doc, "Statements and Declarations", 1)
    statements = [
        (
            "Funding",
            (
                "This work was supported by the National Natural Science Foundation of "
                "China (grant no. 82500006). The funding source had no role in the "
                "study design, data collection, analysis, manuscript preparation, or "
                "the decision to submit for publication."
            ),
        ),
        (
            "Competing interests",
            "The authors declare that they have no competing interests.",
        ),
        (
            "Authors' contributions",
            (
                "All authors contributed to the study. Bofan Wang: conceptualization, "
                "data curation, formal analysis, investigation, methodology, project "
                "administration, resources, software, supervision, validation, "
                "visualization, writing-original draft, and writing-review and editing. "
                "Maoxiang Zhang: data curation, formal analysis, and investigation. "
                "Rufeng Ji: data curation, formal analysis, and investigation. Deyu "
                "Zhao: conceptualization, data curation, formal analysis, "
                "investigation, methodology, project administration, resources, "
                "supervision, validation, and writing-review and editing. Xia Huang: "
                "conceptualization, data curation, formal analysis, funding "
                "acquisition, investigation, methodology, project administration, "
                "resources, supervision, validation, and writing-review and editing. "
                "Deyu Zhao and Xia Huang are the guarantors of this manuscript."
            ),
        ),
        (
            "Acknowledgements",
            (
                "The authors thank the National Natural Science Foundation of China "
                "for funding support (grant no. 82500006)."
            ),
        ),
        (
            "Ethics approval",
            (
                "This study was approved by the Ethics Committee of Children's Hospital "
                "of Nanjing Medical University (approval no. 202601046-1) and was "
                "performed in accordance with the ethical standards of the 1964 "
                "Declaration of Helsinki and its later amendments."
            ),
        ),
        (
            "Consent to participate",
            (
                "Not applicable because of the retrospective design and use of "
                "de-identified data."
            ),
        ),
        ("Consent for publication", "Not applicable."),
        (
            "Data availability",
            (
                "The image dataset analyzed during the current study is not publicly "
                "available because of participant privacy, but it is available from the "
                "corresponding author on reasonable request. Other datasets generated "
                "or analyzed during this study are included in this article and its "
                "Online Resources."
            ),
        ),
        (
            "Code availability",
            (
                "The analysis code and the web-based prediction interface are available "
                "from the corresponding author on reasonable request."
            ),
        ),
    ]
    for label, text in statements:
        add_heading(doc, label, 2)
        add_paragraph(doc, text)

    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.title = f"Title Page - {TITLE}"
    doc.core_properties.subject = "Title page for Pediatric Radiology submission"
    doc.core_properties.comments = ""
    doc.save(OUT / "Title_Page.docx")


def copy_supplementary() -> None:
    mapping = [
        (
            SOURCE_DIR / "Supplementary" / "Supplementary_File_pyradiomics_config.txt",
            SUPPLEMENTARY / "Online_Resource_1_pyradiomics_config.txt",
        ),
        (
            SOURCE_DIR / "Supplementary" / "Supplementary_Fig1_Dice.png",
            SUPPLEMENTARY / "Online_Resource_2_Dice_distribution.png",
        ),
        (
            SOURCE_DIR / "Supplementary" / "Supplementary_Table1_dice_results.xlsx",
            SUPPLEMENTARY / "Online_Resource_3_Dice_results.xlsx",
        ),
        (
            SOURCE_DIR / "Supplementary" / "Supplementary_Table2_ICC_results.xlsx",
            SUPPLEMENTARY / "Online_Resource_4_ICC_results.xlsx",
        ),
        (
            SOURCE_DIR
            / "Supplementary"
            / "Supplementary_Table3_Model_Performance.xlsx",
            SUPPLEMENTARY / "Online_Resource_5_model_performance.xlsx",
        ),
        (
            SOURCE_DIR
            / "Supplementary"
            / "Supplementary_Table4_DeLong_Pvalues_FDRcorrected.xlsx",
            SUPPLEMENTARY / "Online_Resource_6_DeLong_p_values.xlsx",
        ),
    ]
    for source, target in mapping:
        shutil.copy2(source, target)

    for target in SUPPLEMENTARY.glob("*.xlsx"):
        scrub_office_metadata(target)


def scrub_office_metadata(path: Path) -> None:
    """Remove author and descriptive metadata from copied Office Open XML files."""
    empty_tags = (
        "dc:creator",
        "cp:lastModifiedBy",
        "dc:title",
        "dc:subject",
        "dc:description",
        "cp:keywords",
    )
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(path, "r") as source_archive:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as target_archive:
            for item in source_archive.infolist():
                data = source_archive.read(item.filename)
                if item.filename == "docProps/core.xml":
                    text = data.decode("utf-8")
                    for tag in empty_tags:
                        text = re.sub(
                            rf"(<{tag}[^>]*>).*?(</{tag}>)",
                            rf"\1\2",
                            text,
                        )
                    data = text.encode("utf-8")
                target_archive.writestr(item, data)
    temp_path.replace(path)


def write_readme(abstract_words: int, body_words: int) -> None:
    text = f"""# Pediatric Radiology 投稿格式转换说明

## 输出文件

- `Manuscript_PR_anonymized.docx`：匿名主稿，供双盲同行评审使用，已移除作者、单位、邮箱、基金、利益冲突、作者贡献及致谢等身份信息。
- `Title_Page.docx`：单独标题页，含作者、单位、通讯作者、ORCID 待填项、基金、利益冲突、作者贡献和伦理声明。
- `Manuscript_PR_anonymized.md`：正文审阅版，不用于投稿。
- `Figures/`：正文 Fig. 1-Fig. 7，已按 Pediatric Radiology 的 `Fig` + 编号方式命名，并嵌入主稿。
- `Supplementary/`：Online Resource 1-6，已按 Springer 的 Online Resource 引用方式重新命名。

## 依据的 Pediatric Radiology Research 要求

- 双盲评审：主稿和附带文件不出现作者身份信息，另交单独标题页。
- Research 文章正文不超过 18 页双倍行距页，约 4,500 词，图表不计入页数。
- 结构化摘要 150-250 词，分为 Background、Objective、Materials and methods、Results、Conclusion。
- 正文结构为 Introduction、Materials and methods、Results、Discussion、Conclusion。
- Conclusion 控制在 1-2 句，不使用缩写或参考文献。
- 正文使用编号方括号引用及 Springer 数字参考文献格式。
- 页脚自动页码，正文连续行号，最多三级标题。
- 表格题注与脚注分离；表内统计检验标记改为上标小写字母。
- 图题使用加粗 `Fig. 1` 起始格式，图题置于正文文件中，图件同时单独提供。
- 补充材料统一称为 Online Resource，并在正文中按编号引用。

## 已完成的主要修改

1. 摘要重写为 Pediatric Radiology 要求的五段式结构，当前约 {abstract_words} 词。
2. 将结尾总述拆为独立 `Conclusion`，并改为 1-2 句、无缩写、无引用。
3. Introduction-Discussion 当前约 {body_words} 词，低于约 4,500 词限制。
4. 主稿中删除医院名称、伦理批号、通讯邮箱和全部作者身份信息。
5. 保持原始 `Manuscript_EJPwithAPP.docx` 的网页预测界面内容，未使用内容较少的旧版 EJP Markdown。
6. 保留原有 7 幅正文图和 2 张主表；7 幅图已嵌入主稿，并另存为 `Fig1.png`-`Fig7.png`。
7. 补充材料改为 Online Resource 1-6，正文引用已同步更新。
8. 新增自动页码、连续行号和最多三级标题。
9. 图表题注、表注和统计检验脚注按 Springer 格式重排。
10. 已清除 Online Resource 工作簿中的作者和最后修改者元数据。

## 投稿前需要核对

1. `Title_Page.docx` 中 5 位作者的 ORCID iD 仍为 `[to be provided]`。
2. 作者贡献和利益冲突按期刊要求还需在投稿系统界面中再次填写。
3. 原稿 Discussion 提到“from a second hospital”的外部队列，但 Methods 和 Results 中没有该外部验证的方法、样本量或结果。投稿前应补充完整，或删除这一表述。
4. Fig. 1 图片内部写有 `1049 CXR Cases` 和 `59 Radiomics Features`，但正文写的是 931 例、102 个提取特征、98 个可重复特征和 12 个最终模型特征。请确认这些数字分别代表不同筛选阶段；若不一致，应更新图片。
5. 原摘要和原声明对知情同意的描述不完全一致。本版本依据 `Manuscript_EJPwithAPP.docx` 写为 retrospective design 且使用去标识化数据，请确认伦理委员会是否正式豁免知情同意。
6. 期刊鼓励提供 Graphical Abstract，本次未生成；如需要，应按官网 PowerPoint 模板制作。
7. 双盲阶段不要向 Online Resource 文件或图片中补充作者、单位、邮箱等身份信息。

## 原始文件对应关系

- 匿名主稿来源：`European Journal of Pediatrics/Manuscript_EJPwithAPP.docx`
- Fig. 1-3、Fig. 5-7 来源：`European Journal of Pediatrics/Figures/`
- Fig. 4 来源：`European Journal of Pediatrics/Supplementary/Supplementary_Fig4_App_workflow.png`
- Online Resource 1-6 来源：`European Journal of Pediatrics/Supplementary/`
"""
    (OUT / "README_Pediatric_Radiology_格式说明.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUPPLEMENTARY.mkdir(parents=True, exist_ok=True)

    for path in FIGURES.glob("*"):
        if path.is_file():
            path.unlink()
    for path in SUPPLEMENTARY.glob("*"):
        if path.is_file():
            path.unlink()

    src = Document(SOURCE_DOCX)
    md: list[str] = []
    abstract_words, body_words = build_manuscript(src, md)
    build_title_page(abstract_words, body_words)
    copy_supplementary()
    write_readme(abstract_words, body_words)
    (OUT / "Manuscript_PR_anonymized.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Output: {OUT}")
    print(f"Abstract words: {abstract_words}")
    print(f"Body words: {body_words}")
    print(f"Figures: {len(list(FIGURES.glob('*.png')))}")
    print(f"Online resources: {len(list(SUPPLEMENTARY.glob('*')))}")


if __name__ == "__main__":
    main()
