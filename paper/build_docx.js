const fs = require("fs"), path = require("path");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, AlignmentType,
        PageOrientation, Footer, PageNumber, convertInchesToTwip,
        Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle } = D;

const ROOT = __dirname;
const SRC = process.argv[2] || "manuscript.md";
const md = fs.readFileSync(path.join(ROOT, SRC), "utf8").split(/\r?\n/);
const sizes = JSON.parse(fs.readFileSync(path.join(ROOT, ".figsizes.json"), "utf8"));
const FONT = "Times New Roman";

// inline **bold** / *italic* -> TextRun[]
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Courier New", ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out.length ? out : [new TextRun({ text: "", font: FONT })];
}

const body = [];
const P = (children, opts = {}) => body.push(new Paragraph({ children, spacing: { after: 140, line: 300 }, ...opts }));

for (let i = 0; i < md.length; i++) {
  const line = md[i];
  const t = line.trim();
  if (!t || t === "---") continue;

  if (/^\|.*\|$/.test(t)) {                       // markdown pipe table
    const rows = [];
    while (i < md.length && /^\s*\|.*\|\s*$/.test(md[i])) {
      const cells = md[i].trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
      if (!cells.every(c => /^:?-{2,}:?$/.test(c))) rows.push(cells);
      i++;
    }
    i--;
    const nCol = Math.max(...rows.map(r => r.length));
    const TOTAL = 9360;                              // 6.5in in DXA
    const colW = Math.floor(TOTAL / nCol);
    const widths = Array(nCol).fill(colW);
    body.push(new Table({
      columnWidths: widths,
      width: { size: TOTAL, type: WidthType.DXA },
      rows: rows.map((r, ri) => new TableRow({
        tableHeader: ri === 0,
        children: Array.from({ length: nCol }, (_, ci) => new TableCell({
          width: { size: colW, type: WidthType.DXA },
          shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "EDEDED" } : undefined,
          margins: { top: 60, bottom: 60, left: 90, right: 90 },
          children: [new Paragraph({
            spacing: { after: 0, line: 240 },
            children: runs(r[ci] || "", { size: 18, bold: ri === 0 }),
          })],
        })),
      })),
    }));
    body.push(new Paragraph({ text: "", spacing: { after: 120 } }));
    continue;
  }

  const img = t.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
  if (img) {
    const file = path.basename(img[2]);
    const [w, h] = sizes[file];
    const maxW = 540;                       // 5.625in at 96dpi
    const scale = Math.min(1, maxW / w);
    body.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(path.join(ROOT, img[2])),
        transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
      })],
    }));
    continue;
  }

  let m;
  if ((m = t.match(/^#\s+(.*)$/))) {
    P(runs(m[1], { bold: true, size: 30 }), { heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: { after: 200 } });
  } else if ((m = t.match(/^##\s+(.*)$/))) {
    P(runs(m[1], { bold: true, size: 26 }), { heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 } });
  } else if ((m = t.match(/^###\s+(.*)$/))) {
    P(runs(m[1], { bold: true, size: 23 }), { heading: HeadingLevel.HEADING_3, spacing: { before: 220, after: 100 } });
  } else if (/^(Figure|Table) \d+\./.test(t) || /^\*\*Figure \d+\./.test(t)) {
    P(runs(t, { size: 20 }), { alignment: AlignmentType.CENTER, spacing: { after: 240 } });
  } else if ((m = t.match(/^(\d+)\.\s+(.*)$/))) {
    P([new TextRun({ text: m[1] + ".  ", font: FONT, size: 22 }), ...runs(m[2], { size: 22 })],
      { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.35) } });
  } else if ((m = t.match(/^[-*]\s+(.*)$/))) {
    P([new TextRun({ text: "•  ", font: FONT, size: 22 }), ...runs(m[1], { size: 22 })],
      { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.35) } });
  } else {
    const center = /^\*\*David Adekoya\*\*$/.test(t) || /^Federal University/.test(t);
    P(runs(t, { size: 22 }), center ? { alignment: AlignmentType.CENTER, spacing: { after: 80 } }
                                    : { alignment: AlignmentType.JUSTIFIED });
  }
}

const doc = new Document({
  creator: "David Adekoya",
  title: "From Curated Cells to Blood Smears",
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
              margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })],
    })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then(b => {
  const out = path.join(ROOT, (process.argv[3] || "manuscript.docx"));
  fs.writeFileSync(out, b);
  console.log("wrote", out, (b.length / 1024).toFixed(0) + " KB,", body.length, "blocks");
});
