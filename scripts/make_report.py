#!/usr/bin/env python3
"""提出レポート（説明資料）の Word 版を作る。【2026-10-19 新規】
pai_final/ で実行: python3 scripts/make_report.py <出力.docx>
PDF は LibreOffice で変換: soffice --headless --convert-to pdf <出力.docx>
"""
import sys
from docx import Document
from docx.shared import Pt, Mm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = sys.argv[1]
JP = 'Noto Sans CJK JP'
ACC = RGBColor(0xD9, 0x3F, 0x6A); DARK = RGBColor(0x1E, 0x26, 0x3C); SUB = RGBColor(0x5A, 0x63, 0x78)
FIG = 'outputs/report_figs/final/'; SLD = 'outputs/videos/submission_assets/'

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Mm(210), Mm(297)
sec.left_margin = sec.right_margin = Mm(20); sec.top_margin = Mm(18); sec.bottom_margin = Mm(18)

def font(run, size=None, bold=None, color=None):
    run.font.name = JP; run._element.rPr.rFonts.set(qn('w:eastAsia'), JP)
    if size: run.font.size = Pt(size)
    if bold is not None: run.font.bold = bold
    if color is not None: run.font.color.rgb = color
for name in ['Normal', 'Heading 1', 'Heading 2', 'Heading 3', 'Title']:
    st = doc.styles[name]; st.font.name = JP; st.element.rPr.rFonts.set(qn('w:eastAsia'), JP)
doc.styles['Normal'].font.size = Pt(10)
doc.styles['Normal'].paragraph_format.space_after = Pt(3)
doc.styles['Normal'].paragraph_format.line_spacing = 1.25

def H(text, level=1):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(12 if level == 1 else 8); p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text); font(r, 14 if level == 1 else 11.5, True, DARK if level == 1 else ACC)
    if level == 1:  # 下線
        pPr = p._p.get_or_add_pPr(); b = OxmlElement('w:pBdr'); bt = OxmlElement('w:bottom')
        for k, v in (('val', 'single'), ('sz', '8'), ('space', '1'), ('color', 'D93F6A')): bt.set(qn('w:' + k), v)
        b.append(bt); pPr.append(b)
    return p

def P(text, size=10, color=None, bold_prefix=None, after=3):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(after)
    if bold_prefix:
        r = p.add_run(bold_prefix); font(r, size, True, DARK)
    r = p.add_run(text); font(r, size, None, color); return p

def B(text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet'); p.paragraph_format.space_after = Pt(1.5)
    if bold_prefix:
        r = p.add_run(bold_prefix); font(r, 10, True, DARK)
    r = p.add_run(text); font(r, 10); return p

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement('w:shd')
    s.set(qn('w:val'), 'clear'); s.set(qn('w:color'), 'auto'); s.set(qn('w:fill'), hexcolor); tcPr.append(s)

def T(rows, widths=None, header=True, size=8.5, hl=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = ''; p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(str(v)); font(r, size, bool(header and i == 0) or (hl is not None and i == hl), DARK)
            if header and i == 0: shade(c, 'E9ECF3')
            if hl is not None and i == hl: shade(c, 'FFE3EA')
            if widths: c.width = Mm(widths[j])
    if widths:  # LibreOffice は表の格子（tblGrid）の幅を使うので、そちらも固定する
        t.autofit = False
        tblPr = t._tbl.tblPr
        lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
        for gc, w in zip(t._tbl.tblGrid.findall(qn('w:gridCol')), widths): gc.set(qn('w:w'), str(int(w * 56.7)))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def F(path, width_mm, caption):
    doc.add_picture(path, width=Mm(width_mm)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(6)
    r = p.add_run(caption); font(r, 8.5, None, SUB)

# ---------------------------------------------------------------- 表紙ブロック
p = doc.add_paragraph(); r = p.add_run('Physical AI 応用編1　最終課題　説明資料'); font(r, 10, True, ACC)
p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
r = p.add_run('Unitree G1 による TWICE『The Feels』全編ダンス模倣'); font(r, 17, True, DARK)
p = doc.add_paragraph(); r = p.add_run('単眼のダンス動画から、物理シミュレーション上の人型ロボットに 2 分 19 秒のフル尺ダンスを強化学習で真似させる'); font(r, 10.5, None, SUB)
T([['提出動画', 'YouTube（URL は提出フォームに記載）。長さ 2 分 58.9 秒、等倍速（倍速なし）'],
   ['コード', 'GitHub リポジトリ（URL は提出フォームに記載）'],
   ['使用環境', 'Google Colab（A100 80GB）、MuJoCo 物理シミュレーション（実機は使用していない）']],
  widths=[28, 142], header=False, size=9)

# ---------------------------------------------------------------- 1
H('1. 課題の概要と目的')
P('人が踊る様子を撮っただけの普通の動画（単眼カメラ、定点）から全身の動きを取り出し、人型ロボット Unitree G1（29 自由度、身長約 1.3 m、33.3 kg）に、'
  '物理シミュレーションの中で転ばずに同じダンスを最後まで踊らせる。題材は TWICE『The Feels』の踊ってみた動画（伊原六花さん、2 分 19 秒）で、'
  '全編を 1 つのニューラルネット（方策）で踊らせることを目標にした。')
P('ねらいは 2 つある。1 つはモーションキャプチャ機材なしに、ネット上の動画だけから人型ロボットの全身動作を作る流れを、一通り自分の手で組むこと。'
  'もう 1 つは、人とロボットの体格・関節の違いや、映像から推定した動きの誤差という「お手本そのものの不完全さ」に、どこまで対処できるかを確かめることである。')
P('AI の活用：', bold_prefix=None)
B('動画からの人の 3D 姿勢推定に、学習済みモデル GVHMR を使用（推論）。')
B('ロボットの全身制御は、強化学習（PPO）で学習したニューラルネットの方策。4,096 体を GPU 上で並列に動かして学習した。')

# ---------------------------------------------------------------- 2
H('2. システム構成')
F(SLD + 'slide2.png', 165, '図 1　処理の流れ（単眼動画 → 3D 姿勢推定 → リターゲット → お手本の補正 → 物理強化学習）')
T([['段階', '使用したもの', '内容'],
   ['3D 姿勢推定', 'GVHMR', '単眼動画（30fps に変換）から人の 3D 姿勢（SMPL-X、世界座標）を全 4,196 フレーム推定'],
   ['リターゲット', 'GMR', '人の動きを G1 の 29 関節の角度へ変換（IK）'],
   ['お手本の補正', '自作スクリプト', '推定ミスの補間、ゼロ位相フィルタ、接地補正、しゃがみの補正（3 章）。逆動力学による物理チェックで無理な箇所を検出'],
   ['強化学習', 'mjlab + rsl_rl（PPO）', 'Unitree-G1-Tracking タスク。制御 0.02 秒、1 試行最大 10 秒、失敗の多い区間ほど多く練習する適応サンプリング'],
   ['評価・動画', '自作スクリプト', '摩擦を固定して全編を 1 体で通し再生し、誤差・転倒・立ち位置のずれを記録。正面カメラで動画化']],
  widths=[24, 34, 112])

# ---------------------------------------------------------------- 3
H('3. 工夫した点')
H('3-1. 終盤 30 秒の「浮き」の発見と解消', 2)
P('最初の方策は全編を完走したが、110 秒以降だけ位置の誤差が大きかった（110 秒まで平均 7.0 cm、以降 17.9 cm）。関節の角度は合っているのに位置だけずれる点から、'
  'ロボットではなくお手本を疑い、お手本の腰の高さが 110 秒以降だけ約 20 cm 高い（宙に浮いている）ことを突き止めた。位置誤差とお手本の足の高さの相関は r = 0.92。'
  '原因は接地補正の判定（足の高さ 3 cm 未満）で、推定のずれで全身が浮くと接地が 1 フレームも検出されなかった。判定を 25 cm に広げて作り直すと、'
  '同じ方策のままで位置誤差が 42% 下がった（9.32 → 5.41 cm）。')
H('3-2. 物理的に無理な深いしゃがみの補正', 2)
P('9〜14 秒のしゃがみは、人は膝が床に付くほど深いが（腰 0.38 m、膝 157°）、G1 の体格と関節の可動域では届かず、腕どうしのめり込みも起きていた。'
  '区間だけ脚の曲げを 4 割浅くし、腰の高さを補い、めり込む腕は肩を最小限だけ外へ開く補正を自作した（区間の外は 1 フレームも変えない）。'
  '腰の最低の高さは 0.385 → 0.59 m、めり込みは最大 6.7 → 3.8 cm。お手本の補正だけで、この区間の最大誤差が 43% 下がった。')
H('3-3. 報酬の許容幅を段階的に締める追加学習', 2)
P('最初から厳しい報酬で学習すると報酬がほぼ 0 のまま進まないため、まず既定の報酬で「転ばない」段階（約 6,000 反復）まで学習し、その後の追加学習で'
  '誤差の許容幅を段階的に締めた（体の位置 0.3 → 0.25 m、体の向き 0.4 → 0.3 rad、立ち位置 0.3 → 0.2 m、動作の変化への罰則を半分）。'
  '学習済みの重みを引き継ぐので、少ない反復で大きく効いた（4,000 反復で関節誤差 −37%）。立ち位置の許容幅を締めると、最大ドリフトが 31.8 → 16.8 cm に減った。'
  '震え（動作の変化量）は増えていないことも指標を追加して確かめた。')
H('3-4. 評価の落とし穴を見抜き、頑健なモデルを選ぶ', 2)
B('評価だけで「0.02 秒で転倒」と出る問題を、フレームワークのソースを追って特定した（リセット直後に比較用の値がまだ 0 のまま判定していた）。', '評価コードの不具合：')
B('精度を上げた方策が、摩擦 1.0 でだけ止まるようになった。同じ条件でも GPU の計算順の違いで結果が変わる（7 回中 2 回だけ停止）ことを見つけ、1 回の結果で判断せず、'
  '摩擦を 0.5〜1.2 の 6 条件に振り、境界の条件は 4 回ずつ繰り返して評価する方法に変えた。', '精度と頑健性の取り合い：')
B('摩擦への強さを足す追加学習（摩擦の範囲を 0.4〜1.2 に拡大）も試したが、出発点より良くならなかった。最終的に、全 6 条件を完走し精度も最良級の学習途中の版（model_26500）を提出用に選んだ。', 'モデルの選定：')
H('3-5. 見せ方', 2)
P('MuJoCo のカメラの仕様を確かめて正面からの映像にそろえ、人の 3D 推定・お手本・物理のロボットを並べた 3 画面動画で追従の様子を比べられるようにした。'
  '提出動画は、お手本の半透明表示（残像）を消したロボットだけの映像に、元の踊ってみた動画と公式の振付動画を小窓で重ねた。')

# ---------------------------------------------------------------- 4
H('4. 結果')
P('提出用の方策で、2 分 19 秒（139.8 秒）の全編を転倒・停止なしで完走した。誤差は、お手本に対する関節角度の平均の差と、胴体を基準にした体の各部の位置の差（平均・最大）、'
  'および立ち位置の水平のずれ（ドリフト）で測った。')
T([['段階', 'お手本', '関節角度誤差', '位置誤差', '最大位置誤差', '最大ドリフト'],
   ['NB2 ゼロから学習', '元', '0.1635 rad（9.4°）', '9.32 cm', '22.65 cm', '31.2 cm'],
   ['NB3 追加学習 +4,000', '浮き補正', '0.1030 rad', '3.90 cm', '16.70 cm', '23.9 cm'],
   ['NB4 追加学習 +3,000', '＋しゃがみ', '0.0943 rad', '3.50 cm', '9.37 cm', '16.8 cm'],
   ['提出 model_26500', '＋しゃがみ', '0.0934 rad（5.4°）', '3.50 cm', '9.08 cm', '18.5 cm'],
   ['NB2 → 提出', '', '−43%', '−62%', '−60%', '−41%']],
  widths=[44, 22, 34, 20, 26, 24], hl=4)
P('摩擦 0.8、全編。NB2 は 20,000 反復、NB4 の +3,000 反復のうち提出用は 2,502 反復目の版。改善には「お手本の修正」と「学習」の両方が入っている（同じお手本の上で比べた内訳は 3-1・3-2）。'
  '提出動画の書き出し時の評価値は 0.0940 rad / 3.53 cm / 9.07 cm / 17.9 cm で、GPU の計算順による揺れの範囲で一致した。', 8.5, SUB)
F(FIG + 'figA_progress.png', 170, '図 2　段階ごとの追従誤差')
F(FIG + 'figB_friction.png', 150, '図 3　足裏の摩擦を変えたときの完走（提出用の方策は 0.5〜1.2 の 6 条件すべてで完走し、摩擦 1.0 は 6 回とも完走）')
F(FIG + 'figC_timeline.png', 170, '図 4　時刻ごとの誤差（1 秒平均）。しゃがみ区間と、お手本の浮きを直した終盤で、とくに大きく改善した')
F(FIG + 'figD_frames.png', 160, '図 5　3 画面比較動画のコマ（提出用の方策、残像なし）')
P('計算量：学習は Google Colab の A100 で、ゼロからの学習 20,000 反復（約 9 時間）と追加学習（4,000 反復 約 1 時間 47 分、3,000 反復 約 1 時間 22 分）。'
  '1 反復は約 1.6 秒で、方策の上手さによらずほぼ一定だった。', 9)

# ---------------------------------------------------------------- 5
H('5. 考察・限界・今後')
B('関節誤差は最後の段階で −1% しか下がっておらず、評価の揺れと区別できない水準になった。残る誤差の多くはお手本側にあり、学習だけでは下げにくい。', '改善の頭打ち：')
B('しゃがみ（浅くしても体格の差で 0.12 rad 残る）、お手本の足が床の上で動いてしまうフレームが全編の 38%（映像からの推定の誤差で、物理的には正確に真似できない）、'
  '手首どうしのめり込みが最大 3.8 cm 残っている。', '残る誤差の原因：')
B('精度を上げた方策が高い摩擦で止まるようになり、頑健さを足そうとすると精度が落ちた。1 本の振付に合わせ込むこと自体は目的どおりだが、'
  '追い込みすぎると条件の小さな変化に弱くなる。4〜8 秒の方向転換とひねりが、提出用の方策でもいちばん余裕の少ない場面である。', '精度と頑健性：')
B('接地中の足を固定する後処理（foot lock）でお手本の足の滑りを取り除く、腕のめり込みの補正を全編に広げる、実機に向けて外乱や観測の揺らぎを強めた学習を行う。', '今後：')

# ---------------------------------------------------------------- 6
H('6. 動画の構成と出典')
T([['時間', '内容'],
   ['0:00〜2:19.8', 'ロボットのダンス（正面、残像なし）。左下：伊原六花さんの踊ってみた動画（お手本の元）、右下：TWICE 公式の振付動画（参考）。音声は踊ってみた動画のもの'],
   ['2:19.8〜2:58.9', '説明スライド 4 枚（10・10・10・9 秒）：表紙 → アプローチ → 工夫点 → 定量的成果']],
  widths=[30, 140])
P('倍速は使っていない（等倍速）。動画の素材：【定点】TWICE「The Feels」踊ってみた（伊原六花さん）、TWICE "The Feels" Choreography Video (Moving Ver.)。'
  '使用したオープンソース：GVHMR、GMR、mjlab、unitree_rl_mjlab、rsl_rl、MuJoCo。', 9)

doc.save(OUT)
print('saved', OUT)
