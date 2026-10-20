from PIL import Image, ImageDraw, ImageFont, ImageFilter
W,H=1920,1080
B='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'; R='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def f(sz,bold=True): return ImageFont.truetype(B if bold else R, sz, index=0)
BG=(14,18,32); CARD=(28,34,56); ACC=(255,111,145); ACC2=(94,200,255); TXT=(240,242,248); SUB=(160,168,190)
def base(title=None, page=None):
    im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
    # subtle gradient band
    for y in range(H):
        t=y/H; c=tuple(int(BG[i]*(1-t*0.35)+ (40,28,60)[i]*t*0.35) for i in range(3)); d.line([(0,y),(W,y)],fill=c)
    d.rectangle([0,0,W,10],fill=ACC)
    if title:
        d.text((120,90),title,font=f(64),fill=TXT)
        d.rectangle([120,190,280,198],fill=ACC)
    if page: d.text((W-120,H-70),page,font=f(28,False),fill=SUB,anchor='rs')
    d.text((120,H-70),'Physical AI 応用編1 最終課題  |  Unitree G1 × TWICE "The Feels"',font=f(28,False),fill=SUB,anchor='ls')
    return im,d
def rr(d,box,fill,r=28,outline=None,w=0): d.rounded_rectangle(box,r,fill=fill,outline=outline,width=w)
# 【2026-10-19 追加】文字が画面からはみ出していないかの確認（作るたびに表示する）
_orig_text, _orig_mt = ImageDraw.ImageDraw.text, ImageDraw.ImageDraw.multiline_text
BOXES=[]
def _rec(self, xy, text, *a, **k):
    try:
        bb=self.textbbox(xy, text, font=k.get('font'), anchor=k.get('anchor'), spacing=k.get('spacing',4))
        BOXES.append((text.replace('\n','/')[:30], bb, self.im.size))
    except Exception: pass
def _t(self, xy, text, *a, **k): _rec(self, xy, text, *a, **k); return _orig_text(self, xy, text, *a, **k)
def _mt(self, xy, text, *a, **k): _rec(self, xy, text, *a, **k); return _orig_mt(self, xy, text, *a, **k)
ImageDraw.ImageDraw.text, ImageDraw.ImageDraw.multiline_text = _t, _mt

# 1 cover
im,d=base()
st=Image.open('robot_still.png').convert('RGB'); sw=760; st=st.resize((sw,int(st.height*sw/st.width)))
x0,y0=W-sw-120,(H-st.height)//2
mask=Image.new('L',st.size,0); ImageDraw.Draw(mask).rounded_rectangle([0,0,st.width,st.height],30,fill=255)
d.rounded_rectangle([x0-6,y0-6,x0+st.width+6,y0+st.height+6],34,fill=ACC)
im.paste(st,(x0,y0),mask)
d.text((120,250),'最終課題',font=f(34),fill=ACC)
d.text((120,320),'Unitree G1 による',font=f(64),fill=TXT)
d.text((120,410),'TWICE『The Feels』',font=f(84),fill=TXT)
d.text((120,520),'全編ダンス模倣',font=f(84),fill=TXT)
d.multiline_text((120,640),'強化学習（PPO）による\n2分19秒 フル尺モーショントラッキング',font=f(32,False),fill=SUB,spacing=10)
for i,t in enumerate(['GVHMR','GMR','mjlab','PPO']):
    bx=120+i*190; rr(d,[bx,750,bx+170,800+0],CARD,18,outline=ACC2,w=2); d.text((bx+85,775),t,font=f(28),fill=ACC2,anchor='mm')
im.save('slide1.png')

# 2 approach
im,d=base('アプローチ：単眼動画 → 物理シミュレーション上のロボット','2 / 4')
steps=[('1','3D姿勢推定','GVHMR','単眼ダンス動画から\n人体3D姿勢（SMPL）\nを推定'),
       ('2','リターゲット','GMR','人体モーションを\nG1の関節空間へ\n変換'),
       ('3','接地・動作補正','contact / edit','足の接地・浮きや\n自己干渉を補正し\n参照動作を整備'),
       ('4','物理強化学習','mjlab + PPO','物理シミュレーション\n上で参照動作を追従\nする方策を学習')]
cw,gap=370,60; sx=(W-(cw*4+gap*3))//2; y=300
for i,(n,t,tool,desc) in enumerate(steps):
    x=sx+i*(cw+gap); rr(d,[x,y,x+cw,y+520],CARD)
    d.ellipse([x+30,y+30,x+110,y+110],fill=ACC); d.text((x+70,y+70),n,font=f(44),fill=BG,anchor='mm')
    d.text((x+30,y+150),t,font=f(44),fill=TXT)
    rr(d,[x+30,y+225,x+cw-30,y+280],BG,14); d.text((x+cw/2,y+252),tool,font=f(28),fill=ACC2,anchor='mm')
    d.multiline_text((x+30,y+320),desc,font=f(30,False),fill=SUB,spacing=14)
    if i<3:
        ax=x+cw+8; d.polygon([(ax,y+250),(ax+44,y+260),(ax,y+270)],fill=ACC); d.rectangle([ax-10,y+256,ax+10,y+264],fill=ACC)
d.text((W/2,y+600),'入力：伊原六花さん「The Feels」踊ってみた（定点）動画  →  出力：Unitree G1 の全身ダンス方策',font=f(32,False),fill=TXT,anchor='mm')
im.save('slide2.png')

# 3 innovations
im,d=base('工夫点','3 / 4')
items=[('終盤30秒の\n「浮き」解消','contact_h = 0.25 m','接地判定の高さを調整し、\n終盤で足が浮いて\n追従が崩れる問題を解消'),
       ('最難所しゃがみの\n自己干渉補正','edit_reference','深いしゃがみで生じる\n脚・腕の自己干渉を\n参照動作の編集で補正'),
       ('効率的な\n追加学習','fine-tuning','既存方策を初期値に\n追加学習し、再学習の\nコストを大幅に削減')]
cw,gap=500,70; sx=(W-(cw*3+gap*2))//2; y=290
for i,(t,k,desc) in enumerate(items):
    x=sx+i*(cw+gap); rr(d,[x,y,x+cw,y+560],CARD); d.rectangle([x+40,y+40,x+120,y+48],fill=ACC2)
    d.multiline_text((x+40,y+80),t,font=f(42),fill=TXT,spacing=10)
    rr(d,[x+40,y+250,x+cw-40,y+315],BG,14); d.text((x+cw/2,y+282),k,font=f(32),fill=ACC,anchor='mm')
    d.multiline_text((x+40,y+360),desc,font=f(31,False),fill=SUB,spacing=14)
im.save('slide3.png')

# 4 metrics
im,d=base('定量的成果','4 / 4')
# 【2026-10-19 更新】提出用の方策 model_26500（摩擦 0.8、全編 139.8 秒）
mets=[('0.093','rad','関節角度誤差（平均）','約 5.4°'),('3.5','cm','位置誤差（平均）',''),
      ('9.1','cm','最大位置誤差','10 cm 未満を達成'),('18.5','cm','最大ドリフト','')]
cw,ch,gx,gy=780,230,60,40; sx=(W-(cw*2+gx))//2; y0=260
for i,(v,u,l,n) in enumerate(mets):
    x=sx+(i%2)*(cw+gx); y=y0+(i//2)*(ch+gy); rr(d,[x,y,x+cw,y+ch],CARD)
    d.text((x+50,y+40),l,font=f(34),fill=TXT)
    fv=f(110); d.text((x+50,y+200),v,font=fv,fill=ACC2,anchor='ls'); vw=d.textlength(v,font=fv)
    d.text((x+64+vw,y+200),u,font=f(46),fill=ACC2,anchor='ls')
    if n: rr(d,[x+cw-50-d.textlength(n,font=f(30))-40,y+140,x+cw-50,y+200],BG,14); d.text((x+cw-70,y+170),n,font=f(30),fill=ACC,anchor='rm')
by=y0+2*(ch+gy)+10; rr(d,[sx,by,sx+cw*2+gx,by+120],ACC)
BANNER='2分19秒 全編を完走 ｜ 摩擦 0.5〜1.2 の全6条件で転倒なし'
bs=50
while d.textlength(BANNER,font=f(bs))>cw*2+gx-80: bs-=2
d.text((W/2,by+60),BANNER,font=f(bs),fill=BG,anchor='mm')
im.save('slide4.png')

# overlays for part1 (1280x720 canvas)
o=Image.new('RGBA',(1280,720),(0,0,0,0)); d=ImageDraw.Draw(o)
pw,ph=400,225; px,py=1280-pw-24,720-ph-24
d.rounded_rectangle([px-4,py-4,px+pw+4,py+ph+4],8,outline=(255,255,255,235),width=4)
d.rounded_rectangle([px-4,py-40,px+190,py-4],8,fill=(255,111,145,230)); d.text((px+8,py-22),'TWICE 公式 (参考)',font=f(20),fill=(255,255,255),anchor='lm')
d.rounded_rectangle([20,20,420,64],10,fill=(0,0,0,150)); d.text((36,42),'Unitree G1 ｜ 強化学習ポリシー (PPO)',font=f(22),fill=(255,255,255),anchor='lm')
o.save('overlay.png')

bad=[(t,bb) for t,bb,(w,h) in BOXES if bb[0]<0 or bb[1]<0 or bb[2]>w or bb[3]>h]
print('文字の数', len(BOXES), '/ 画面からはみ出した文字:', bad or 'なし')
