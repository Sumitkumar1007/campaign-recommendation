from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUT = Path("artifacts/reports/campaign_recommendation_stakeholder_overview.pptx")
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def emu(inches: float) -> int:
    return int(inches * 914400)


def color(hex_value: str) -> str:
    return hex_value.replace("#", "").upper()


def text_box(
    shape_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    size: int = 24,
    bold: bool = False,
    fill: str | None = None,
    line: str | None = None,
    font_color: str = "1F2937",
    radius: bool = False,
) -> str:
    paragraphs = []
    for raw_line in text.split("\n"):
        if raw_line == "":
            paragraphs.append("<a:p/>")
            continue
        lines = raw_line.split("||")
        runs = []
        for index, part in enumerate(lines):
            runs.append(
                f"""
                <a:r>
                  <a:rPr lang="en-US" sz="{size * 100}"{' b="1"' if bold else ''}>
                    <a:solidFill><a:srgbClr val="{color(font_color)}"/></a:solidFill>
                  </a:rPr>
                  <a:t>{esc(part)}</a:t>
                </a:r>
                """
            )
            if index < len(lines) - 1:
                runs.append("<a:br/>")
        paragraphs.append(f"<a:p>{''.join(runs)}</a:p>")

    fill_xml = (
        f"<a:solidFill><a:srgbClr val=\"{color(fill)}\"/></a:solidFill>"
        if fill
        else "<a:noFill/>"
    )
    line_xml = (
        f"<a:ln w=\"12700\"><a:solidFill><a:srgbClr val=\"{color(line)}\"/></a:solidFill></a:ln>"
        if line
        else "<a:ln><a:noFill/></a:ln>"
    )
    preset = "roundRect" if radius else "rect"
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
        <a:prstGeom prst="{preset}"><a:avLst/></a:prstGeom>
        {fill_xml}
        {line_xml}
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" bIns="45720"/>
        <a:lstStyle/>
        {''.join(paragraphs)}
      </p:txBody>
    </p:sp>
    """


def title_slide(title: str, subtitle: str) -> str:
    shapes = [
        text_box(2, 0.75, 1.25, 11.9, 1.0, title, 40, True, font_color="0F172A"),
        text_box(3, 0.8, 2.35, 10.8, 1.0, subtitle, 22, False, font_color="475569"),
        text_box(4, 0.8, 5.6, 11.0, 0.55, "Stakeholder overview | simple technical explanation", 16, False, font_color="64748B"),
    ]
    return slide_xml(shapes)


def content_slide(title: str, bullets: list[str], footnote: str | None = None) -> str:
    body = "\n".join(f"• {item}" for item in bullets)
    shapes = [
        text_box(2, 0.55, 0.35, 12.2, 0.6, title, 28, True, font_color="0F172A"),
        text_box(3, 0.85, 1.25, 11.6, 4.8, body, 20, False, font_color="1F2937"),
    ]
    if footnote:
        shapes.append(text_box(4, 0.85, 6.25, 11.4, 0.45, footnote, 14, False, font_color="64748B"))
    return slide_xml(shapes)


def flow_slide() -> str:
    steps = [
        ("Raw data", "Communication + case records"),
        ("Clean data", "Valid EMI window rows"),
        ("Build features", "3-month account history"),
        ("Model predict", "Best day/channel/time"),
        ("Apply rules", "Risk-based top 1/2/3"),
        ("Store output", "DB tables + CSV summary"),
    ]
    shapes = [text_box(2, 0.55, 0.35, 12.2, 0.6, "Production Prediction Flow", 28, True, font_color="0F172A")]
    x = 0.55
    for index, (head, body) in enumerate(steps, start=3):
        shapes.append(text_box(index, x, 1.55, 1.82, 1.15, f"{head}\n{body}", 14, True, fill="E0F2FE", line="38BDF8", font_color="0F172A", radius=True))
        if index < 8:
            shapes.append(text_box(index + 20, x + 1.78, 1.92, 0.35, 0.35, "→", 22, True, font_color="0EA5E9"))
        x += 2.08
    shapes.append(text_box(30, 0.85, 4.05, 11.4, 1.4, "Plain meaning: we use recent customer communication behavior to decide next month’s recommended contact plan.", 22, False, fill="F8FAFC", line="CBD5E1", font_color="334155", radius=True))
    return slide_xml(shapes)


def training_slide() -> str:
    rows = [
        ("1", "Collect history", "Past communication rows from Postgres"),
        ("2", "Create labels", "Successful channel-time-language patterns"),
        ("3", "Create features", "Per-account 3-month behavior summary"),
        ("4", "Train model", "CatBoost learns patterns per EMI-relative day"),
        ("5", "Validate", "Check performance on later month data"),
    ]
    shapes = [text_box(2, 0.55, 0.35, 12.2, 0.6, "How Model Is Trained", 28, True, font_color="0F172A")]
    y = 1.2
    for sid, head, body in rows:
        shapes.append(text_box(10 + int(sid), 0.85, y, 0.55, 0.5, sid, 18, True, fill="DCFCE7", line="22C55E", font_color="166534", radius=True))
        shapes.append(text_box(20 + int(sid), 1.55, y - 0.03, 10.5, 0.62, f"{head}: {body}", 19, False, font_color="1F2937"))
        y += 0.9
    return slide_xml(shapes)


def slide_xml(shapes: list[str]) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def rels_xml(target: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="{target}"/>
</Relationships>"""


def build() -> None:
    slides = [
        title_slide("Campaign Recommendation Project", "How model is trained, how prediction runs, and what business output is created"),
        content_slide("Project Purpose", [
            "Recommend EMI communication plan for each loan account.",
            "Decide day, channel, time, and language for D-5 to D+5.",
            "Keep D as no communication.",
            "Control contact intensity by customer risk bucket.",
        ]),
        content_slide("Business Output", [
            "One row per loan account for next prediction month.",
            "Daily columns: D-5, D-4, D-3, D-2, D-1, D, D+1 ... D+5.",
            "Strategy looks like SMS-9AM-ENGLISH or WH-5PM-HINDI.",
            "Blank '-' means no recommendation for that day.",
        ]),
        content_slide("Risk Rule", [
            "LOW risk: top 1 recommendation per day.",
            "MEDIUM risk: top 2 recommendations per day.",
            "HIGH risk: top 3 recommendations per day.",
            "This keeps business control outside model complexity.",
        ]),
        training_slide(),
        content_slide("What Model Learns From", [
            "Latest 3 months of account-level communication history.",
            "SMS, WhatsApp, and IVR/Voice activity intensity.",
            "Past successful send time and language patterns.",
            "Customer risk and source month context.",
            "Only valid EMI-window days are used.",
        ]),
        content_slide("What Success Means", [
            "Success means useful communication signal, not direct payment conversion.",
            "SMS/WhatsApp success uses delivered, read, clicked style statuses.",
            "Voice/IVR success uses connected call statuses.",
            "Model learns which strategy historically worked for similar accounts.",
        ]),
        flow_slide(),
        content_slide("Monthly Production Run", [
            "Fetch current source month data from Postgres.",
            "Fetch previous months needed for 3-month history.",
            "Prepare features and schedule-shaped data.",
            "Load CatBoost production model.",
            "Predict next month account schedules.",
            "Store results back to Postgres and save summary files.",
        ]),
        content_slide("Airflow Monthly Inference DAG", [
            "Airflow will schedule the monthly prediction job.",
            "Typical run: after monthly source data is available.",
            "DAG checks data availability before prediction starts.",
            "If input rows are missing, job stops safely and logs reason.",
            "Successful run stores predictions, campaign rows, mappings, and audit counts.",
        ]),
        content_slide("Monthly Inference DAG Steps", [
            "Task 1: load configuration and month parameters.",
            "Task 2: fetch communication history and current cases.",
            "Task 3: build model-ready features.",
            "Task 4: run CatBoost prediction.",
            "Task 5: store output tables in Postgres.",
            "Task 6: write audit summary and notify team.",
        ]),
        content_slide("Retraining Pipeline", [
            "Retraining is separate from monthly prediction.",
            "It runs when new labeled history is ready or model quality drops.",
            "Pipeline rebuilds training data from latest historical months.",
            "Model is trained and validated before promotion.",
            "Only approved model is moved to production use.",
        ]),
        content_slide("Airflow Retraining DAG", [
            "Airflow will also schedule or trigger retraining.",
            "DAG creates training dataset, schedule labels, and monthly features.",
            "DAG trains CatBoost model and saves metrics.",
            "Validation report is reviewed before model promotion.",
            "Production inference DAG then uses promoted model bundle or MLflow alias.",
        ]),
        content_slide("Latest Run Example", [
            "Source month: APR-2026.",
            "Prediction month: MAY-2026.",
            "Case rows processed: 82,187.",
            "Model-scored rows with history: 2,192.",
            "Prediction snapshots stored: 82,187.",
            "Campaign scheduler rows stored: 392.",
            "Campaign mapping rows stored: 33,354.",
        ], "Rows without enough history are safely returned as blank recommendations."),
        content_slide("Where Output Goes", [
            "Account-level predictions table: ai_ml_recommendations_data.",
            "Feature snapshot table: ai_ml_recommendations_feature.",
            "Campaign scheduler table: ai_ml_campaign_recommendations.",
            "Campaign mapping table: ai_ml_campaign_mapping.",
            "Audit table records run status and counts.",
        ]),
        content_slide("Simple Takeaway", [
            "Model converts past communication behavior into next-month campaign guidance.",
            "Business rules keep recommendation count controlled by risk.",
            "Airflow will automate both monthly inference and controlled retraining.",
            "Production pipeline stores both predictions and campaign rows.",
            "Main improvement area: reduce blank recommendations by increasing usable account history/features.",
        ]),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(slides)))
        zf.writestr("_rels/.rels", package_rels())
        zf.writestr("docProps/core.xml", core_props())
        zf.writestr("docProps/app.xml", app_props(len(slides)))
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        zf.writestr("ppt/theme/theme1.xml", theme_xml())
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels())
        for idx, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide)
            zf.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", rels_xml("../slideLayouts/slideLayout1.xml"))
    print(OUT)


def content_types(n: int) -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, n + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>"""


def package_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def core_props() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Campaign Recommendation Stakeholder Overview</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-05-14T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-05-14T00:00:00Z</dcterms:modified>
</cp:coreProperties>"""


def app_props(n: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{n}</Slides>
</Properties>"""


def presentation_xml(n: int) -> str:
    slide_ids = "\n".join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, n + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{n+1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def presentation_rels(n: int) -> str:
    rels = [
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, n + 1)
    ]
    rels.append(f'<Relationship Id="rId{n+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
    rels.append(f'<Relationship Id="rId{n+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Stakeholder">
  <a:themeElements>
    <a:clrScheme name="Stakeholder"><a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="0284C7"/></a:accent1><a:accent2><a:srgbClr val="16A34A"/></a:accent2><a:accent3><a:srgbClr val="F59E0B"/></a:accent3><a:accent4><a:srgbClr val="DC2626"/></a:accent4><a:accent5><a:srgbClr val="7C3AED"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Default"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def slide_master_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def slide_master_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def slide_layout_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def slide_layout_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


if __name__ == "__main__":
    build()
