/* Mi trayectoria: the CV as a pop-up pixel book.
   Each chapter is a small pixel-art room standing on one long street of a 3D Santiago
   (Three.js). Scrolling walks Diego from room to room: the page he leaves folds down,
   the next one pops up, and the camera pulls back to show the road in between.
   Every texture is drawn by hand on small canvases with the same palette and 3 × 5
   alphabet as the home world (life-scene.js), and redrawn when the theme changes. */

const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';
const journey = document.getElementById('journey');
const stage = document.getElementById('tj-stage');
const view = document.getElementById('tj-canvas');
const chapterEls = [...document.querySelectorAll('.tj-chapter')];
const cards = chapterEls.map(el => el.querySelector('.tj-card'));
const motion = matchMedia('(prefers-reduced-motion: reduce)');

/* ---------- Pixel kit ---------- */
let c;
const colors = {
  ink:'#33463e', dark:'#253d35', cream:'#f3eed9', paper:'#e6e4d0',
  green:'#8caa5e', leaf:'#64865a', lime:'#c9df85', rust:'#bc7352',
  skin:'#bf8c63', skinLight:'#dbac7f', hair:'#293631', white:'#f9f5df',
  blue:'#537c8e', navy:'#3b5870', terracotta:'#c78968', wood:'#ae9067',
  gold:'#e2b84f', goldDark:'#b38a35', screen:'#2d4840', board:'#466853', steel:'#9aa5a0'
};
const palettes = {
  day:{sky:'#acd0d2', skyLow:'#d4e2d3', haze:'#d4e2d3', mountain:'#94acb5', mountainDark:'#78959e', snow:'#f2f0de', hill:'#8ca886', hillLight:'#acbd89', city:'#bac6b3', cityWindow:'#d4d7c2', wall:'#e7e2ce', room:'#f0ecd9', roomShade:'#dce0c6', frame:'#829483', pillar:'#c1c5af', floor:'#b9a382', floorEdge:'#e4ddbe', road:'#58655e', roadLine:'#d5d9b1', walk:'#c78968', walkLine:'#dfb28b', curb:'#e4ddbe', grass:'#b0bf8b', grassDark:'#95ad77', grassLight:'#c5cf9c', glass:'#b9d8cd', lamp:'#e6e6ca', tree:'#64865a', treeLight:'#8caa5e', cloud:'#edf0df', star:'#e4e5bd'},
  dusk:{sky:'#c4aba9', skyLow:'#ead1b3', haze:'#e2cbb4', mountain:'#888c9c', mountainDark:'#717c8c', snow:'#eaded5', hill:'#748671', hillLight:'#93a080', city:'#aba999', cityWindow:'#efd29a', wall:'#d9cdb7', room:'#ded4bd', roomShade:'#c9c7ac', frame:'#7a8778', pillar:'#b5b5a2', floor:'#a99273', floorEdge:'#dccfb3', road:'#555f5b', roadLine:'#d9c9a3', walk:'#b97c62', walkLine:'#d4a283', curb:'#dccfb3', grass:'#9fae80', grassDark:'#879c6c', grassLight:'#b6bf91', glass:'#f6d59b', lamp:'#f4d48a', tree:'#5a7a55', treeLight:'#7f9b5c', cloud:'#f1dcc7', star:'#fff0bf'},
  night:{sky:'#172839', skyLow:'#263c4d', haze:'#22384a', mountain:'#3b5366', mountainDark:'#2d4353', snow:'#8da6b3', hill:'#344f48', hillLight:'#4c6655', city:'#34494c', cityWindow:'#d9bd7b', wall:'#a19c80', room:'#d4c59c', roomShade:'#a7ac82', frame:'#5c6e60', pillar:'#8c8f7c', floor:'#8a7a62', floorEdge:'#a59b80', road:'#2b3c40', roadLine:'#8d9478', walk:'#7e5c4e', walkLine:'#9a7562', curb:'#8f8a74', grass:'#3f5747', grassDark:'#344a3d', grassLight:'#4f6a52', glass:'#f6d59b', lamp:'#f4d48a', tree:'#2f4a3d', treeLight:'#41604b', cloud:'#2c4459', star:'#e4e5bd'}
};
let mode = 'day', p = palettes.day;

function R(x,y,w,h,color) { c.fillStyle=color; c.fillRect(Math.round(x),Math.round(y),Math.round(w),Math.round(h)); }
function line(x0,y0,x1,y1,color,width=1) {
  x0=Math.round(x0);y0=Math.round(y0);x1=Math.round(x1);y1=Math.round(y1);
  const dx=Math.abs(x1-x0), sx=x0<x1?1:-1, dy=-Math.abs(y1-y0), sy=y0<y1?1:-1;
  let error=dx+dy;
  while(true) { R(x0,y0,width,width,color);if(x0===x1&&y0===y1)break;const twice=2*error;if(twice>=dy){error+=dy;x0+=sx;}if(twice<=dx){error+=dx;y0+=sy;} }
}
function poly(points,color) {
  const min=Math.ceil(Math.min(...points.map(v=>v[1]))), max=Math.floor(Math.max(...points.map(v=>v[1])));
  for(let y=min;y<=max;y++) {
    const hits=[];
    for(let i=0,j=points.length-1;i<points.length;j=i++) {
      const a=points[i],b=points[j];
      if((a[1]<=y&&b[1]>y)||(b[1]<=y&&a[1]>y))hits.push(a[0]+(y-a[1])/(b[1]-a[1])*(b[0]-a[0]));
    }
    hits.sort((a,b)=>a-b);
    for(let i=0;i<hits.length;i+=2)R(Math.ceil(hits[i]),y,Math.floor(hits[i+1])-Math.ceil(hits[i])+1,1,color);
  }
}
function circle(x,y,r,color) { for(let dy=-r;dy<=r;dy++){const dx=Math.floor(Math.sqrt(r*r-dy*dy));R(x-dx,y+dy,dx*2+1,1,color);} }
function outline(x,y,w,h,color) { R(x,y,w,1,color);R(x,y+h-1,w,1,color);R(x,y,1,h,color);R(x+w-1,y,1,h,color); }
function dither(x,y,w,h,color,phase=0) { for(let j=0;j<h;j++)for(let i=(j+phase)%2;i<w;i+=2)R(x+i,y+j,1,1,color); }
const glyphs = {
 A:'010101111101101',B:'110101110101110',C:'011100100100011',D:'110101101101110',E:'111100110100111',F:'111100110100100',G:'011100101101011',H:'101101111101101',I:'111010010010111',J:'001001001101010',K:'101101110101101',L:'100100100100111',M:'101111111101101',N:'101111111111101',O:'010101101101010',P:'110101110100100',Q:'010101101111011',R:'110101110101101',S:'011100010001110',T:'111010010010010',U:'101101101101111',V:'101101101101010',W:'101101111111101',X:'101101010101101',Y:'101101010010010',Z:'111001010100111',
 '0':'111101101101111','1':'010110010010111','2':'110001010100111','3':'110001010001110','4':'101101111001001','5':'111100110001110','6':'011100111101111','7':'111001010010010','8':'111101111101111','9':'111101111001110',
 ' ':'000000000000000','.':'000000000000010',':':'000010000010000','/':'001001010100100','-':'000000111000000','+':'000010111010000','=':'000111000111000','>':'100010001010100','<':'001010100010001','!':'010010010000010','?':'110001010000010','&':'010101010101011','(':'010100100100010',')':'010001001001010','_':'000000000000111','%':'101001010100101',',':'000000000010100','#':'101111101111101','*':'000101010101000'
};
function text(value,x,y,color=colors.ink,scale=1) {
  [...value.toUpperCase().normalize('NFD').replace(/[̀-ͯ]/g,'')].forEach((ch,index)=>{
    const bits=glyphs[ch]||glyphs[' '];
    for(let k=0;k<15;k++)if(bits[k]==='1')R(x+index*4*scale+(k%3)*scale,y+Math.floor(k/3)*scale,scale,scale,color);
  });
}
function sign(label,x,y,w,color=colors.ink,ink=colors.cream) { R(x,y,w,12,color);text(label,x+Math.floor((w-label.length*4+1)/2),y+4,ink); }
function plant(x,y,size=1) { R(x-3,y,7,5,colors.rust);R(x-1,y-9*size,2,9*size,colors.leaf);R(x-6,y-7*size,5,4*size,colors.leaf);R(x+1,y-10*size,5,5*size,colors.green); }
function books(x,y,n=7) { for(let i=0;i<n;i++){const h=6+(i*7%6);R(x+i*4,y-h,3,h,[colors.rust,colors.blue,colors.green,colors.wood][i%4]);R(x+i*4,y-3,3,1,colors.cream);} }
function monitor(x,y,w=27,h=18) { R(x,y,w,h,colors.ink);R(x+2,y+2,w-4,h-5,colors.screen);R(x+w/2-1,y+h,3,4,colors.ink);R(x+w/2-6,y+h+4,13,2,colors.ink); }
function rand(seed) { let s=seed%2147483647; if(s<=0)s+=2147483646; return ()=>((s=s*16807%2147483647)-1)/2147483646; }
const smooth = (a,b,x) => { const t=Math.min(1,Math.max(0,(x-a)/(b-a))); return t*t*(3-2*t); };
const lerp = (a,b,t) => a+(b-a)*t;
const backOut = x => { const k=1.9; return 1+(k+1)*Math.pow(x-1,3)+k*Math.pow(x-1,2); };

/* ---------- Diego: one sprite, many poses (drawn facing right) ---------- */
const DW=48, DH=46;
function head(x,y,{blink=false,cap=null}={}) {
  R(x+1,y,5,2,colors.hair);R(x,y+2,7,3,colors.hair);
  R(x+1,y+3,6,5,colors.skin);R(x+4,y+3,3,4,colors.skinLight);R(x+6,y+5,2,2,colors.skinLight);
  R(x+1,y+4,1,2,colors.hair);R(x+5,y+4,1,1,blink?colors.skin:colors.ink);
  R(x+3,y+8,3,2,colors.skin);
  if(cap){R(x,y,7,2,cap);R(x+5,y+1,4,1,cap);}
}
function legs(x,F) { R(x+1,F-9,3,7,colors.navy);R(x+5,F-9,3,7,colors.navy);R(x,F-2,4,2,colors.white);R(x+5,F-2,5,2,colors.white); }
function torso(x,F,shirt) { R(x,F-19,8,10,shirt);R(x+1,F-10,7,1,colors.ink); }
function backArm(x,F,shirt) { R(x-1,F-19,2,3,shirt);R(x-1,F-16,2,6,colors.skin); }
function frontArm(x,F,shirt) { R(x+7,F-19,2,3,shirt);R(x+7,F-16,2,6,colors.skin); }
function standing(x,F,shirt,blink,{front=true,headOpts={}}={}) { backArm(x,F,shirt);legs(x,F);torso(x,F,shirt);head(x,F-29,{blink,...headOpts});if(front)frontArm(x,F,shirt); }
const poses = {
  stand(x,F,t,o) { standing(x,F,o.shirt,o.blink); },
  walk(x,F,t,o) {
    const sw=[3,0,-3,0][o.step], b=o.step%2?-1:0;
    line(x+3,F-10,x+3-sw,F-3,colors.navy,2);R(x+2-sw,F-2,4,2,colors.white);
    line(x,F-17+b,x+sw,F-11+b,colors.skin,2);
    R(x-4,F-18+b,4,8,colors.rust);R(x-4,F-18+b,4,1,'#9c5c40');
    R(x,F-19+b,8,9,o.shirt);R(x+1,F-10,7,1,colors.ink);R(x+1,F-19+b,1,6,'#9c5c40');
    head(x,F-29+b,{blink:o.blink});
    line(x+5,F-10,x+5+sw,F-3,colors.navy,2);R(x+4+sw,F-2,5,2,colors.white);
    line(x+6,F-17+b,x+6-sw,F-11+b,colors.skin,2);
  },
  wave(x,F,t,o) {
    standing(x,F,o.shirt,o.blink,{front:false});
    const w=Math.floor(t*4)%2;
    R(x+7,F-19,2,3,o.shirt);line(x+8,F-17,x+11,F-22,colors.skin,2);line(x+11,F-22,x+11+w*2,F-28,colors.skin,2);R(x+10+w*2,F-30,3,3,colors.skinLight);
  },
  read(x,F,t,o) {
    standing(x,F,o.shirt,o.blink,{front:false});
    R(x+6,F-16,4,2,colors.skin);
    R(x+8,F-19,9,6,colors.cream);R(x+12,F-19,1,6,'#bdb79c');
    for(let i=0;i<3;i++){R(x+9,F-17+i*2,2,1,'#9aa893');R(x+14,F-17+i*2,2,1,'#9aa893');}
    if(Math.floor(t*.7)%3===0)R(x+13,F-21,4,6,colors.white);
    R(x+15,F-15,2,2,colors.skinLight);
  },
  grad(x,F,t,o) {
    const lift=Math.round(Math.max(0,Math.sin(t*5))*3);F-=lift;
    legs(x,F);R(x-1,F-19,10,13,colors.ink);R(x+3,F-19,2,4,colors.cream);
    head(x,F-29,{blink:o.blink});
    R(x-2,F-31,12,2,colors.ink);R(x+1,F-29,7,1,colors.ink);
    const tassel=Math.floor(t*4)%2;R(x+9,F-30,1,3+tassel,colors.gold);
    line(x+7,F-17,x+11,F-24,colors.skin,2);
    R(x+8,F-30,9,3,colors.cream);R(x+12,F-30,1,3,colors.rust);
  },
  sit(x,F,t,o) {
    R(x-4,F-20,2,13,colors.ink);R(x-4,F-7,12,2,colors.ink);R(x-3,F-5,1,5,colors.ink);R(x+6,F-5,1,5,colors.ink);
    R(x+1,F-10,9,3,colors.navy);R(x+7,F-8,3,6,colors.navy);R(x+7,F-2,6,2,colors.white);
    R(x,F-17,8,8,o.shirt);
    head(x,F-27,{blink:o.blink});
    R(x+5,F-16,4,2,o.shirt);R(x+8,F-15,6,2,colors.skin);R(x+13,F-15+(Math.floor(t*6)%2),3,2,colors.skinLight);
  },
  point(x,F,t,o) {
    standing(x,F,o.shirt,o.blink,{front:false});
    const r=Math.floor(t*2)%2;
    R(x+7,F-19,2,3,o.shirt);line(x+8,F-17,x+15,F-21-r,colors.skin,2);R(x+16,F-22-r,2,1,colors.skinLight);
  },
  teach(x,F,t,o) {
    standing(x,F,o.shirt,o.blink,{front:false});
    const s=Math.sin(t*1.6)*3;
    R(x+7,F-19,2,3,o.shirt);line(x+8,F-17,x+12,F-19,colors.skin,2);
    line(x+13,F-19,x+26,F-29+s,colors.wood);
    R(x+8,F-12,2,1,colors.skin);
  },
  coffee(x,F,t,o) {
    standing(x,F,o.shirt,o.blink,{front:false});
    const sip=(t%4)>3.1, my=sip?-24:-18;
    R(x+7,F-19,2,3,o.shirt);line(x+8,F-16,x+10,F-12,colors.skin,2);line(x+10,F-12,x+10,F+my+4,colors.skin,2);
    R(x+9,F+my,5,5,colors.cream);R(x+14,F+my+1,1,3,colors.cream);
    if(!sip)for(let i=0;i<2;i++)R(x+10+i*2,F+my-3-((Math.floor(t*3)+i)%3),1,2,'#c6c3ad');
  },
  trophy(x,F,t,o) {
    const lift=Math.round(Math.max(0,Math.sin(t*4))*3);F-=lift;
    legs(x,F);torso(x,F,o.shirt);head(x,F-29,{blink:o.blink});
    R(x-1,F-19,2,3,o.shirt);R(x+7,F-19,2,3,o.shirt);
    line(x,F-19,x-1,F-30,colors.skin,2);line(x+8,F-19,x+9,F-30,colors.skin,2);
    R(x-1,F-41,11,6,colors.gold);R(x-3,F-40,2,4,colors.gold);R(x+10,F-40,2,4,colors.gold);
    R(x+1,F-41,2,4,'#f6dc8a');R(x+3,F-35,3,2,colors.goldDark);R(x+1,F-33,7,2,colors.goldDark);
  },
  build(x,F,t,o) {
    standing(x,F,o.shirt,o.blink,{front:false});
    const k=Math.floor(t*5)%3, a=[-.9,-.3,.35][k];
    R(x+7,F-19,2,3,o.shirt);line(x+8,F-17,x+12,F-14,colors.skin,2);
    const hx=x+13, hy=F-14;line(hx,hy,hx+Math.cos(a)*9,hy+Math.sin(a)*9,colors.steel,2);
    R(hx+Math.cos(a)*9-1,hy+Math.sin(a)*9-1,4,3,colors.steel);
  }
};

/* ---------- Rooms (256 × 88; floor at row 84, Diego's feet stand on row 84) ---------- */
const SW=256, SH=88, FLOOR=84, GAP=384;
function floorSlab() { R(0,FLOOR,SW,SH-FLOOR,p.floor);R(0,FLOOR,SW,1,p.floorEdge); }
function room(label,signColor,signInk=colors.cream) {
  const w=label.length*4+9, x=Math.round((SW-w)/2);
  R(x+5,12,2,8,colors.ink);R(x+w-7,12,2,8,colors.ink);
  sign(label,x,0,w,signColor,signInk);
  R(2,19,SW-4,4,colors.cream);R(2,23,SW-4,2,'#788e7b');
  R(4,25,SW-8,FLOOR-25,p.frame);R(6,25,SW-12,FLOOR-25,p.wall);R(10,27,SW-20,FLOOR-27,p.room);R(10,FLOOR-12,SW-20,12,p.roomShade);
  R(4,25,5,FLOOR-25,p.pillar);R(SW-9,25,5,FLOOR-25,p.pillar);
  floorSlab();
}
function desk(x,w,top=69) { R(x,top,w,3,colors.wood);R(x+2,top+3,2,FLOOR-top-3,colors.wood);R(x+w-4,top+3,2,FLOOR-top-3,colors.wood); }
function student(x,y,i,t,raise=false) {
  const hair=['#604c38','#455149','#7d6e4b','#2f3a35'][i%4], shirt=[colors.blue,colors.leaf,colors.rust,colors.navy][i%4];
  const bob=Math.floor(t*1.3+i)%3===0?1:0;
  R(x,y+bob,5,4,hair);R(x+1,y+3+bob,4,3,colors.skinLight);R(x-1,y+6,7,7,shirt);
  if(raise){R(x+5,y-5,2,11,colors.skinLight);}
}
function coworker(x,F,t,hair,shirt) {
  // A seated colleague (not Diego): different hair, no glasses.
  R(x-4,F-20,2,13,colors.ink);R(x-4,F-7,12,2,colors.ink);R(x-3,F-5,1,5,colors.ink);R(x+6,F-5,1,5,colors.ink);
  R(x+1,F-10,9,3,colors.ink);R(x+7,F-8,3,6,colors.ink);R(x+7,F-2,6,2,colors.white);
  R(x,F-17,8,8,shirt);
  R(x,F-27,7,5,hair);R(x+1,F-24,6,5,colors.skinLight);R(x,F-25,2,5,hair);R(x+5,F-23,1,1,colors.ink);R(x+3,F-19,3,2,colors.skinLight);
  R(x+5,F-16,4,2,shirt);R(x+8,F-15,6,2,colors.skinLight);R(x+13,F-15+(Math.floor(t*5+1)%2),3,2,colors.skinLight);
}
const worldMap=[
 '..........##.........##########.....',
 '.....########.....##.#############..',
 '...##########....###############....',
 '....########.....##############.....',
 '.....######.......############......',
 '......####.......#..##########......',
 '.......##.......######..######......',
 '........###.....#######...###.......',
 '........####.....#####.....#........',
 '.........###......###.......##......',
 '.........##.......##.......####.....',
 '..........#.................##......'];
const mapCells=[];worldMap.forEach((row,y)=>[...row].forEach((ch,x)=>{if(ch==='#')mapCells.push([x,y]);}));

const SCENES = {
  casa: {
    spot:150, pose:'wave', facing:1, shirt:colors.blue,
    base() {
      room('SANTIAGO',colors.ink,colors.lime);
      R(20,32,42,28,colors.wood);R(22,34,38,24,p.glass);
      poly([[22,57],[30,46],[35,51],[43,39],[52,50],[56,46],[60,52],[60,57]],mode==='night'?'#3b5366':'#94acb5');
      poly([[40,43],[43,39],[46,43],[43,42]],p.snow);R(22,45,38,1,colors.wood);R(40,34,2,24,colors.wood);
      R(14,64,4,20,colors.wood);R(18,72,48,5,colors.wood);R(19,77,2,7,colors.wood);R(63,77,2,7,colors.wood);
      R(18,68,48,4,colors.cream);R(36,67,30,5,colors.blue);R(36,67,30,1,'#6c94a6');R(20,65,11,4,colors.white);
      R(76,34,20,25,colors.cream);outline(76,34,20,25,colors.wood);text('42K',80,38,colors.rust);
      poly([[78,56],[84,46],[88,51],[94,56]],colors.green);
      R(102,46,30,2,colors.wood);books(103,46,7);
      R(70,81,6,3,colors.lime);R(77,81,6,3,colors.lime);R(70,83,13,1,colors.ink);
      desk(176,52,69);R(186,50,28,19,'#cfc8ad');R(189,52,22,13,colors.screen);R(184,66,32,3,'#bdb79c');R(196,69,8,0,colors.ink);
      R(220,60,8,9,colors.rust);R(221,58,6,2,colors.cream);
      plant(240,79);
    },
    anim(t) {
      text('HOLA',191,55,colors.lime);if(Math.floor(t*2)%2)R(208,55,3,5,colors.lime);
      R(191,61,4+Math.floor(t*3)%12,1,'#82b59c');
    }
  },
  uni: {
    spot:128, pose:'read', facing:1, shirt:colors.green,
    base() {
      room('USACH',colors.navy);
      R(16,31,94,34,colors.wood);R(18,33,90,29,colors.board);
      text('MAX Z = 3X+2Y',22,37,'#d7e5ae');text('S.A. X+Y<4',22,45,colors.cream);text('X,Y>0',22,53,colors.cream);
      R(80,45,1,14,colors.cream);R(80,58,24,1,colors.cream);poly([[81,57],[81,47],[96,57]],'#82b59c');R(92,49,2,2,colors.rust);
      R(18,62,90,2,colors.wood);R(30,61,5,1,colors.white);R(40,61,3,1,'#e6a77f');
      circle(140,39,6,colors.ink);circle(140,39,5,colors.cream);
      R(210,32,30,22,colors.cream);outline(210,32,30,22,colors.wood);text('PIE',219,36,colors.blue);for(let i=0;i<3;i++)R(214,44+i*3,22-i*5,1,'#9aa893');
    },
    anim(t) {
      const a=t*2.4, b=t*.2;
      line(140,39,140+Math.round(Math.cos(a)*4),39+Math.round(Math.sin(a)*4),colors.rust);
      line(140,39,140+Math.round(Math.cos(b)*3),39+Math.round(Math.sin(b)*3),colors.ink);
      for(let i=0;i<3;i++){const x=166+i*28;student(x+4,58,i,t,i===1&&Math.floor(t*.5)%3===0);desk(x-2,22,70);R(x+2,66,10,3,colors.navy);}
      if(Math.floor(t*3)%4===0)R(103,58,1,1,colors.white);
    }
  },
  msc: {
    spot:140, pose:'grad', facing:1, shirt:colors.ink,
    base() {
      room('MAGISTER',colors.leaf);
      R(16,31,76,32,'#c9cdbd');R(18,33,72,28,'#eef0e2');
      const layers=[[26,[38,47,56]],[44,[36,43,50,57]],[62,[39,47,55]],[80,[43,51]]];
      for(let l=0;l<3;l++)for(const ya of layers[l][1])for(const yb of layers[l+1][1])line(layers[l][0],ya,layers[l+1][0],yb,'#b5bfae');
      layers.forEach(([x,ys],l)=>ys.forEach(y=>{circle(x,y,2,[colors.blue,colors.green,colors.green,colors.rust][l]);}));
      for(let x=100;x<220;x+=8){poly([[x,28],[x+6,28],[x+3,33]],[colors.lime,colors.rust,colors.blue,colors.cream][(x/8)%4]);}R(100,28,120,1,colors.ink);
      R(98,70,16,14,colors.cream);for(let y=72;y<84;y+=3)R(98,y,16,1,'#d3cdb2');R(100,64,13,6,colors.white);R(101,60,11,4,colors.cream);
      R(160,38,20,16,colors.wood);R(162,40,16,12,colors.cream);text('MSC',164,43,colors.ink);R(167,49,6,2,colors.rust);
      desk(186,36,69);R(196,61,15,8,colors.ink);R(197,62,13,6,colors.screen);R(194,68,19,1,'#9aa893');
      R(228,44,16,40,colors.dark);for(let y=48;y<80;y+=4)R(231,y,10,1,'#4f675c');
    },
    anim(t) {
      for(let i=0;i<3;i++)R(199,63+i*2,3+((i*5+Math.floor(t*3))%8),1,'#82b59c');
      for(let i=0;i<5;i++)if((Math.floor(t*3)+i)%3)R(239,49+i*6,2,1,colors.lime);
      if(!motion.matches)for(let i=0;i<22;i++){const x=104+(i*37)%110, y=30+((t*16+i*23)%54), w=Math.floor(t*6+i)%2;R(x+(w?1:0),y,2,w?1:2,[colors.lime,colors.rust,colors.blue,colors.gold,colors.cream][i%5]);}
    }
  },
  falabella: {
    spot:168, pose:'sit', facing:1, shirt:colors.blue,
    base() {
      room('FALABELLA','#839c59');
      R(14,31,82,36,colors.ink);R(16,33,78,32,colors.screen);text('FORECAST',19,36,'#82b59c');
      R(19,44,1,18,'#4f675c');R(19,61,72,1,'#4f675c');text('ERR -15%',58,36,colors.lime);
      R(104,38,32,7,colors.white);R(108,34,13,5,colors.white);R(120,31,10,8,colors.white);R(130,36,8,8,colors.white);text('GCP',111,39,colors.blue);
      R(102,52,9,7,colors.blue);R(118,52,9,7,colors.green);R(134,52,9,7,colors.rust);R(111,55,7,1,colors.ink);R(127,55,7,1,colors.ink);
      line(120,46,106,51,colors.ink);line(120,46,122,51,colors.ink);line(120,46,138,51,colors.ink);
      R(206,31,40,30,colors.wood);R(208,33,36,26,colors.cream);
      for(let i=0;i<9;i++)R(210+(i*7)%32,35+(i*5)%22,1,1,'#c9c3a6');
      desk(172,40,69);monitor(184,47,22,16);
      plant(152,79,.8);
    },
    anim(t) {
      let old=55;
      for(let i=0;i<34;i++){const x=20+i, y=Math.round(52-Math.sin(i*.55)*4-i*.12+Math.sin(i*1.7)*1.5);line(x-1,old,x,y,colors.cream);old=y;}
      const reveal=Math.floor((t*8)%40);
      for(let i=0;i<Math.min(reveal,36);i++){const x=54+i, y=Math.round(52-Math.sin((i+34)*.55)*4-(i+34)*.12);R(x,y,1,1,colors.lime);if(i%3===0)R(x,y-3,1,1,'#82b59c');}
      const dot=(t*1.2)%1;R(102+dot*40,50,2,1,colors.lime);
      const pts=[[212,55],[220,40],[232,46],[240,37],[236,54]];
      for(let i=0;i<pts.length;i++){const [x0,y0]=pts[i],[x1,y1]=pts[(i+1)%pts.length];line(x0,y0,x1,y1,'#a2b5a6');circle(x0,y0,1,colors.rust);}
      const seg=(t*.6)%pts.length, si=Math.floor(seg), f=seg-si, a=pts[si], b=pts[(si+1)%pts.length];
      R(a[0]+(b[0]-a[0])*f-1,a[1]+(b[1]-a[1])*f-1,3,3,colors.blue);
      for(let i=0;i<3;i++)R(188,50+i*3,4+((i*5+Math.floor(t*4))%12),1,['#a5cbb7','#cbda8c','#d9c295'][i]);
    }
  },
  walmart: {
    spot:150, pose:'point', facing:-1, shirt:colors.cream,
    base() {
      room('WALMART','#537f96');
      R(14,31,108,53,colors.wood);R(16,33,104,49,'#d9d3b8');
      const clusters=[colors.blue,colors.rust,colors.green];
      for(const y of [46,62,78]){R(16,y,104,2,colors.wood);
        for(let i=0;i<13;i++){const x=18+i*8, h=6+((i*7+y)%5), k=Math.min(2,Math.floor(i/4.4));R(x,y-h,6,h,clusters[k]);R(x+1,y-h+1,4,1,'#ffffff55');}}
      R(166,31,76,34,colors.ink);R(168,33,72,30,colors.screen);text('CAUSAL',172,36,'#82b59c');
      R(172,61,64,1,'#4f675c');text('A',179,55,colors.cream);text('B',203,55,colors.cream);
      R(208,72,26,9,colors.steel);for(let x=211;x<232;x+=4)R(x,73,1,7,'#c5cdc8');line(234,72,238,64,colors.ink);R(237,63,4,2,colors.ink);
      circle(212,82,2,colors.ink);circle(230,82,2,colors.ink);
    },
    anim(t) {
      const k=Math.floor(t*1.5)%3;
      const span=[[16,49],[50,83],[84,119]][k];
      outline(span[0],31,span[1]-span[0]+2,52,colors.lime);outline(span[0]-1,30,span[1]-span[0]+4,54,colors.ink);
      const g=Math.min(1,(t%3)/1.5);
      R(184,44,9,15,'#82b59c');R(212,59-Math.round(19*g),9,Math.round(19*g),colors.lime);
      if(g>=1){line(223,44,229,38,colors.lime);R(228,38,2,1,colors.lime);R(229,39,1,2,colors.lime);}
    }
  },
  profe: {
    spot:108, pose:'teach', facing:1, shirt:colors.rust,
    base() {
      room('USACH / APPLIED ML',colors.board,'#d7e5ae');
      R(126,31,110,36,colors.wood);R(128,33,106,31,colors.board);
      text('APPLIED ML',132,37,'#d7e5ae');text('Y = F(X)',132,46,colors.cream);
      R(190,38,1,22,colors.cream);R(190,59,40,1,colors.cream);
      const r=rand(11);for(let i=0;i<14;i++){const x=193+r()*34, y=40+r()*17, cls=(x-190)*0.55-(59-y)>-4;R(x,y,2,2,cls?colors.lime:'#e6a77f');}
      R(128,64,106,2,colors.wood);
    },
    anim(t) {
      const prog=smooth(0,1,(t%5)/3);
      line(192,58,192+36,58-Math.round(lerp(4,20,prog)),colors.cream);
      for(let i=0;i<3;i++){const x=20+i*28;student(x+6,58,i+1,t,i===2&&Math.floor(t*.6)%3===1);desk(x,24,70);R(x+12,64,10,6,colors.ink);R(x+13,65,8,4,mode==='night'?'#bfe0c8':'#9fcdb6');}
    }
  },
  visa: {
    spot:150, pose:'coffee', facing:-1, shirt:colors.navy,
    base() {
      room('VISA','#506c80');
      R(14,31,116,42,'#22384d');R(16,33,112,38,'#2a4660');
      for(const [x,y] of mapCells)R(18+x*3,35+y*3,2,2,'#5f86a1');
      text('ANALYTICS & CONSULTING',16,76,'#7c8974');
      desk(166,72,69);monitor(186,48,22,15);monitor(212,48,22,15);plant(244,79,.8);
    },
    anim(t) {
      const r=rand(Math.floor(t*1.2)+3);
      for(let i=0;i<5;i++){const [x,y]=mapCells[Math.floor(r()*mapCells.length)];const ph=(t*1.2)%1;R(18+x*3,35+y*3,2,2,colors.lime);if(ph<.5&&i<2){R(17+x*3,34+y*3,4,1,'#c9df8588');}}
      const a=mapCells[(Math.floor(t*.8)*7)%mapCells.length], b=mapCells[(Math.floor(t*.8)*13+40)%mapCells.length];
      const f=(t*.8)%1;for(let s=0;s<=f*10;s++){const k=s/10;const x=lerp(a[0],b[0],k)*3+19, y=lerp(a[1],b[1],k)*3+36-Math.sin(k*Math.PI)*8;R(x,y,1,1,colors.cream);}
      coworker(170,FLOOR,t,'#8a6a45',colors.green);
      for(let i=0;i<3;i++){R(190,52+i*3,4+((i*3+Math.floor(t*3))%12),1,'#a5cbb7');R(216+i*5,60-i*2-(Math.floor(t*2+i)%3),3,3+i*2,colors.green);}
    }
  },
  logros: {
    spot:128, pose:'trophy', facing:1, shirt:colors.blue,
    base() {
      room('LOGROS',colors.rust);
      const shine=mode==='night'?'#f6e3a6':'#fbf6e2';
      for(const x of [40,128,214])dither(x-14,48,28,36,shine,x%2);
      text('MEJOR EGRESADO',14,32,'#7c8974');text('2022',30,38,'#7c8974');
      R(28,62,24,22,colors.cream);R(26,60,28,3,colors.wood);
      text('HACKATHON',110,32,'#7c8974');text('IA POR EL FUTURO',96,38,'#7c8974');
      R(188,31,52,34,colors.wood);R(190,33,48,30,colors.cream);outline(193,36,42,24,colors.gold);
      text('ML ENGINEER',192,40,colors.ink);text('GCP 2023',198,48,colors.blue);circle(214,57,3,colors.gold);
      R(196,66,36,18,colors.cream);R(194,64,40,3,colors.wood);
    },
    anim(t) {
      const sw=Math.round(Math.sin(t*2)*1);
      R(39+sw,44,2,8,colors.rust);R(41+sw,44,2,8,colors.blue);circle(41+sw,54,4,colors.gold);circle(41+sw,54,2,'#f6dc8a');
      R(206,58,16,6,colors.blue);R(210,54,8,5,colors.lime);
      const r=rand(Math.floor(t*3)+1);
      for(let i=0;i<5;i++){const x=20+r()*216, y=30+r()*40;R(x,y-1,1,3,colors.white);R(x-1,y,3,1,colors.white);}
    }
  },
  taller: {
    spot:70, pose:'build', facing:1, shirt:colors.leaf,
    base() {
      room('TALLER',colors.wood);
      R(14,31,46,32,'#cbb98f');for(let y=34;y<62;y+=4)for(let x=17;x<58;x+=4)R(x,y,1,1,'#a89772');
      R(20,36,2,14,colors.steel);R(18,36,6,3,colors.steel);
      R(30,36,8,4,colors.ink);R(33,40,2,14,colors.wood);
      R(44,36,2,10,colors.rust);R(44,46,2,6,colors.steel);
      R(50,38,6,6,colors.steel);R(52,40,2,2,'#cbb98f');
      desk(78,46,67);
      R(96,56,14,11,colors.steel);R(98,58,10,2,'#c5cdc8');R(98,47,10,8,'#b8c3bd');R(102,43,2,4,colors.ink);R(101,42,4,2,colors.rust);
      const rows=[[FLOOR,['PYTHON','PYTORCH','DOCKER']],[FLOOR-12,['SQL','GCP','AWS','TF','HF']],[FLOOR-24,['FASTAPI','FLASK','WEB']]];
      const crate=[colors.wood,colors.terracotta,colors.green,colors.blue];
      rows.forEach(([bottom,labels],r)=>{let x=134+r*6;labels.forEach((label,i)=>{const w=label.length*4+5;R(x,bottom-11,w,11,crate[(i+r)%4]);R(x+1,bottom-8,w-2,7,colors.cream);text(label,x+3,bottom-7,colors.ink);x+=w+2;});});
    },
    anim(t) {
      const blink=Math.floor(t*2.2)%5===0;
      R(100,49,2,2,blink?'#b8c3bd':colors.lime);R(104,49,2,2,blink?'#b8c3bd':colors.lime);R(100,53,6,1,colors.ink);
      if(Math.floor(t*5)%3===2&&!motion.matches){R(92,60,1,1,colors.gold);R(90,57,1,1,colors.lime);R(94,58,1,1,colors.white);}
    }
  },
  epilogo: {
    spot:94, pose:'wave', facing:1, shirt:colors.lime,
    base() {
      R(0,FLOOR,SW,SH-FLOOR,p.walk);R(0,FLOOR,SW,1,p.walkLine);
      R(56,18,2,FLOOR-18,colors.ink);R(132,18,2,FLOOR-18,colors.ink);
      R(56,18,78,12,colors.ink);text('CONTINUARA...',69,22,colors.lime);
      for(let x=58;x<132;x+=8)dither(x,30,4,3,colors.cream,0);
      R(176,22,4,FLOOR-22,colors.wood);
      const arrows=[['BLOG',colors.cream,colors.ink,26],['LINKEDIN','#537c8e',colors.cream,40],['GITHUB',colors.ink,colors.lime,54]];
      for(const [label,bg,ink,y] of arrows){const w=label.length*4+10;R(178,y,w,10,bg);poly([[178+w,y],[183+w,y+5],[178+w,y+9]],bg);text(label,183,y+3,ink);}
      for(const wx of [204,229]){circle(wx,76,7,colors.ink);circle(wx,76,5,p.walk);R(wx,76,1,1,colors.cream);}
      line(204,76,213,66,colors.rust,2);line(213,66,217,76,colors.rust,2);line(217,76,204,76,colors.rust);line(213,66,225,66,colors.rust);line(225,66,217,76,colors.rust);line(224,62,229,76,colors.cream);R(210,63,7,2,colors.ink);R(222,61,5,1,colors.ink);
      R(14,68,26,3,colors.wood);R(14,63,26,3,colors.wood);R(16,71,2,13,colors.wood);R(36,71,2,13,colors.wood);
    },
    anim(t) {
      const f=Math.floor(t*3)%3;
      R(150,30,1,54,colors.ink);
      R(151,30,14,4,colors.white);R(151,34,14,4,'#bd6655');R(151,30,5,4,'#52718a');R(152,31,2,2,colors.white);R(165,31+f,4,3,colors.white);R(165,34+f,4,3,'#bd6655');
    }
  }
};

/* ---------- Background art: long strips drawn once per theme ---------- */
function drawFar(ctx,w,h) {
  c=ctx;c.clearRect(0,0,w,h);
  const r=rand(17), pts=[[0,h]], peaks=[];
  let x=-20;
  while(x<w+40){pts.push([x,h-18-r()*22]);x+=16+r()*26;const py=h-(32+r()*(h-40));pts.push([x,py]);peaks.push([x,py]);x+=16+r()*26;}
  pts.push([w,h]);
  poly(pts,p.mountain);
  c.globalCompositeOperation='source-atop';
  for(const [px,py] of peaks)if(py<h-62)poly([[px-34,py+18],[px-16,py+11],[px-8,py+17],[px+3,py+10],[px+13,py+16],[px+34,py+9],[px+34,py-4],[px-34,py-4]],p.snow);
  c.globalCompositeOperation='source-over';
}
let farLayer, midLayer;
function drawMid(ctx,w,h,x0,k) {
  c=ctx;c.clearRect(0,0,w,h);
  const toPx=wx=>Math.round((wx-x0)/k);
  const r=rand(5), pts=[[0,h]];
  for(let x=0;x<=w+30;x+=26+Math.floor(r()*24))pts.push([x,h-26-r()*26]);
  pts.push([w,h]);poly(pts,p.mountainDark);
  // Cerro San Cristóbal with its statue, near the university chapters.
  const sx=toPx(GAP*1.5);
  poly([[sx-90,h],[sx-40,h-36],[sx-12,h-58],[sx+8,h-60],[sx+30,h-40],[sx+90,h]],p.hill);
  poly([[sx-60,h],[sx-22,h-34],[sx-6,h-52],[sx+4,h-50],[sx-8,h-20],[sx+10,h]],p.hillLight);
  R(sx-3,h-68,7,9,colors.cream);R(sx-1,h-74,3,6,colors.cream);R(sx,h-77,1,3,colors.cream);
  // City blocks, with the Costanera tower near today.
  for(let x=0,i=0;x<w;i++){const bw=10+Math.floor(r()*12), bh=6+Math.floor(r()*20);R(x,h-bh,bw,bh,p.city);for(let wy=h-bh+3;wy<h-2;wy+=4)for(let wx=x+2;wx<x+bw-2;wx+=3)if(r()>.45)R(wx,wy,1,1,p.cityWindow);x+=bw+1+Math.floor(r()*3);}
  const tx=toPx(GAP*6.4);
  poly([[tx-6,h],[tx-4,h-70],[tx,h-76],[tx+4,h-70],[tx+6,h]],mode==='night'?'#4b6466':'#bdc9be');
  for(let y=h-66;y<h-2;y+=4)R(tx-4,y,9,1,mode==='night'?'#d9bd7b':'#8fa8a1');
}
function drawTree(ctx,kind) {
  c=ctx;c.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
  const w=ctx.canvas.width,h=ctx.canvas.height;
  if(kind==='round'){R(w/2-2,h-16,4,16,colors.wood);R(w/2-12,h-40,24,16,p.tree);R(w/2-16,h-33,32,10,p.tree);R(w/2-7,h-45,15,12,p.treeLight);R(w/2-13,h-33,14,6,p.treeLight);}
  else if(kind==='poplar'){R(w/2-1,h-8,3,8,colors.wood);R(w/2-5,h-44,10,38,p.tree);R(w/2-3,h-50,6,8,p.tree);R(w/2-3,h-40,4,24,p.treeLight);}
  else {R(4,h-12,w-8,12,p.tree);R(8,h-17,w-18,6,p.treeLight);R(w-14,h-15,8,6,p.tree);}
}
function drawLamp(ctx,year) {
  c=ctx;c.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
  const x=6;
  R(x,4,2,58,colors.ink);R(x-3,2,8,2,colors.ink);R(x-2,4,6,4,mode==='day'?p.lamp:'#ffe7a8');R(x-3,8,8,1,colors.ink);
  if(mode!=='day')dither(x-4,9,10,6,'#f8d184',0);
  const w=year.length*4+7;R(x+2,22,w,11,colors.cream);outline(x+2,22,w,11,colors.ink);text(year,x+6,25,colors.ink);
  R(x-2,62,6,2,colors.ink);
}
function drawBus(ctx) {
  c=ctx;c.clearRect(0,0,64,24);
  R(2,4,58,15,'#c16b57');R(4,1,52,3,'#e9e3d2');for(let i=0;i<6;i++)R(6+i*8,6,6,6,mode==='night'?'#f2d38f':'#acccbf');
  R(2,14,58,3,'#e6debe');circle(12,19,3,colors.ink);circle(48,19,3,colors.ink);R(56,13,3,2,'#f0d997');
}
/* Street sport: Diego (sunglasses, lime) runs or rides by, most of the time with her. */
const HER={hair:'#4a3328', shirt:'#d9826f', skin:'#c99670', skinLight:'#e2b48c', bike:'#537c8e'};
function sportHead(x,y,{her=false,helmet=false,bob=0}={}) {
  const hair=her?HER.hair:colors.hair, skin=her?HER.skin:colors.skin, light=her?HER.skinLight:colors.skinLight;
  R(x+1,y,5,2,hair);R(x,y+2,7,3,hair);
  R(x+1,y+3,6,5,skin);R(x+4,y+3,3,4,light);R(x+6,y+5,2,2,light);
  R(x+1,y+4,1,2,hair);R(x+3,y+8,3,2,skin);
  if(her){R(x,y+2,2,6,hair);R(x-3,y+2-bob,3,2,hair);R(x-4,y+3-bob,2,4,hair);R(x+5,y+4,1,1,colors.ink);R(x+5,y+7,1,1,'#b9655a');}
  else {R(x+3,y+3,5,2,colors.dark);R(x+4,y+3,1,1,'#738e85');}
  if(helmet){R(x,y,7,2,her?HER.shirt:colors.lime);R(x+5,y+1,4,1,her?HER.shirt:colors.lime);}
}
function runner(x,F,t,her=false) {
  const step=Math.floor(t*7)%4, bob=step%2, swing=[4,0,-4,0][step], y=F-28;
  const shirt=her?HER.shirt:colors.lime, skin=her?HER.skin:colors.skin;
  sportHead(x,y+bob,{her,bob});
  R(x,y+9+bob,8,7,shirt);R(x+1,y+16+bob,7,3,her?colors.ink:colors.navy);
  line(x+2,y+19,x+2+swing,y+24,skin,2);line(x+2+swing,y+24,x+5+swing,y+26,skin,2);R(x+4+swing,y+26,5,2,colors.white);
  line(x+6,y+19,x+6-swing,y+23,skin,2);line(x+6-swing,y+23,x+4-swing,y+26,colors.ink,2);R(x+3-swing,y+26,5,2,colors.white);
  const a=Math.max(0,swing/2);
  line(x+1,y+10,x-3,y+14+a,skin,2);line(x-3,y+14+a,x-5+swing/2,y+11,skin,2);
  line(x+7,y+10,x+11,y+14-a,skin,2);line(x+11,y+14-a,x+13-swing/2,y+10,skin,2);
}
function cyclist(x,F,t,her=false) {
  const y=F-24, frame=her?HER.bike:colors.rust, shirt=her?HER.shirt:colors.lime, skin=her?HER.skin:colors.skin, light=her?HER.skinLight:colors.skinLight;
  for(const wx of [x,x+25]){circle(wx,y+15,8,colors.ink);circle(wx,y+15,6,p.road);line(wx-5*Math.cos(t*9),y+15-5*Math.sin(t*9),wx+5*Math.cos(t*9),y+15+5*Math.sin(t*9),'#bac8ab');R(wx,y+15,2,1,colors.cream);}
  line(x,y+15,x+9,y+4,frame,2);line(x+9,y+4,x+14,y+15,frame,2);line(x+14,y+15,x,y+15,frame);line(x+9,y+4,x+22,y+4,frame);line(x+22,y+4,x+14,y+15,frame);
  line(x+20,y,x+25,y+15,colors.cream);line(x+20,y,x+25,y-1,colors.ink);R(x+6,y+1,7,2,colors.ink);
  sportHead(x+13,y-17,{her,helmet:true});line(x+15,y-7,x+9,y+1,shirt,5);line(x+16,y-6,x+22,y-3,skin,2);line(x+22,y-3,x+24,y,skin,2);
  const a=t*8+(her?1.6:0), px=x+14+Math.cos(a)*4, py=y+8+Math.sin(a)*3;
  line(x+10,y+1,px,py,her?colors.ink:colors.navy,3);line(px,py,px,py+7,light,2);R(px-1,py+7,5,2,colors.white);
}
// Canvases hold the pair side by side; she rides a little behind him.
const RUN_W=52, RUN_H=32, BIKE_W=100, BIKE_H=44;
function drawSport(ctx,kind,together,t) {
  c=ctx;c.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
  if(kind==='run'){const F=RUN_H-1;if(together){runner(8,F,t+.37,true);runner(32,F,t);}else runner(20,F,t);}
  else {const F=BIKE_H-1;if(together){cyclist(10,F,t+.2,true);cyclist(60,F,t);}else cyclist(36,F,t);}
}
function drawTuft(ctx,i) {
  c=ctx;c.clearRect(0,0,16,12);
  R(3,6,2,6,p.grassDark);R(6,3,2,9,p.grassDark);R(9,5,2,7,p.grassDark);R(12,7,2,5,p.grassDark);
  if(i===1){R(5,1,3,3,mode==='night'?'#cfc6a0':colors.cream);R(6,2,1,1,colors.gold);}
  if(i===2){R(10,3,3,3,colors.rust);}
}
function drawWalk(ctx) { c=ctx;const w=32,h=28;R(0,0,w,h,p.walk);R(0,0,w,2,p.floorEdge);for(let x=0;x<w;x+=16)R(x,2,1,h-4,p.walkLine);R(0,14,w,1,p.walkLine);R(0,h-3,w,3,p.curb); }
function drawRoad(ctx) { c=ctx;R(0,0,32,46,p.road);R(0,22,16,2,p.roadLine);R(0,2,32,1,'#ffffff18');R(0,43,32,3,p.grassDark); }
function drawGrass(ctx) { c=ctx;R(0,0,64,64,p.grass);const r=rand(9);for(let i=0;i<40;i++)R(Math.floor(r()*64),Math.floor(r()*64),2,1,r()>.5?p.grassDark:p.grassLight); }
function drawSky(ctx,w,h,t) {
  c=ctx;
  const bands=[p.sky,p.sky,p.sky,mixHex(p.sky,p.skyLow,.35),mixHex(p.sky,p.skyLow,.7),p.skyLow,p.haze];
  const bh=Math.ceil(h/bands.length);
  bands.forEach((col,i)=>{R(0,i*bh,w,bh+1,col);if(i>0)dither(0,i*bh,w,2,bands[i-1],0);});
  if(mode==='night'){
    const r=rand(3);for(let i=0;i<Math.floor(w*h/700);i++){const x=Math.floor(r()*w), y=Math.floor(r()*h*.62);R(x,y,1,1,Math.sin(t*.8+i*2.1)>.35?p.star:'#788e9c');if(i%17===0){R(x-1,y,3,1,'#a5bbc1');R(x,y-1,1,3,'#a5bbc1');}}
    const mx=Math.round(w*.8), my=Math.round(h*.18);circle(mx,my,9,'#a6bdbe');circle(mx,my,7,'#f4edbd');circle(mx+4,my-3,7,p.sky);
  } else {
    const sx=Math.round(w*(mode==='dusk'?.74:.8)), sy=Math.round(h*(mode==='dusk'?.42:.16));
    circle(sx,sy,10,mode==='dusk'?'#f1c17f':'#ece7b0');circle(sx,sy,7,mode==='dusk'?'#ffe0a3':'#fff4c4');
    for(let i=0;i<Math.ceil(w/70);i++){const x=Math.round(((i*137+t*(1.2+i%4*.3))%(w+60))-30), y=12+(i*37%Math.round(h*.38));R(x,y,30,5,p.cloud);R(x+6,y-4,19,5,p.cloud);R(x+11,y-7,9,4,p.cloud);}
  }
}
function mixHex(a,b,t) { const pa=parseInt(a.slice(1),16), pb=parseInt(b.slice(1),16); const ch=s=>Math.round(lerp((pa>>s)&255,(pb>>s)&255,t)); return '#'+((ch(16)<<16)|(ch(8)<<8)|ch(0)).toString(16).padStart(6,'0'); }

/* ---------- Boot: plain list stays as the fallback if Three.js or WebGL is missing ---------- */
function hasWebGL() { try { const test=document.createElement('canvas'); return Boolean(test.getContext('webgl2')||test.getContext('webgl')); } catch { return false; } }
let THREE = null;
if (journey && hasWebGL()) { try { THREE = await import(THREE_URL); } catch { THREE = null; } }
if (THREE) start();

function start() {
  const keys = chapterEls.map(el => el.dataset.scene);
  const N = keys.length;
  const renderer = new THREE.WebGLRenderer({canvas:view, antialias:false, powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 10, 9000);
  scene.add(camera);
  scene.fog = new THREE.Fog(0xffffff, 1300, 4200);
  const textures = [];
  const canvasOf = (w,h) => { const cv=document.createElement('canvas'); cv.width=w; cv.height=h; return cv; };
  function texture(cv) {
    const tx=new THREE.CanvasTexture(cv);
    tx.magFilter=THREE.NearestFilter;tx.minFilter=THREE.NearestFilter;tx.generateMipmaps=false;tx.colorSpace=THREE.SRGBColorSpace;
    textures.push(tx);return tx;
  }
  function billboard(cv,w,h,{fog=true,anchor='bottom'}={}) {
    const geo=new THREE.PlaneGeometry(w,h);if(anchor==='bottom')geo.translate(0,h/2,0);
    const mat=new THREE.MeshBasicMaterial({map:texture(cv),alphaTest:.5,side:THREE.DoubleSide,fog});
    return new THREE.Mesh(geo,mat);
  }
  const redrawers = [];
  const xMin=-1800, xMax=(N-1)*GAP+1800, length=xMax-xMin;

  // Sky: a camera-attached backdrop, so it never parallaxes.
  const skyCanvas=canvasOf(16,120), skyCtx=skyCanvas.getContext('2d');
  const sky=billboard(skyCanvas,1,1,{fog:false,anchor:'center'});
  sky.material.depthWrite=false;sky.renderOrder=-1;sky.position.z=-8000;camera.add(sky);

  // Long parallax strips (split into tiles that every GPU accepts).
  function strip(z,k,heightPx,draw) {
    // One art pixel covers `unit` world units, so pixels look the same size as the rooms'.
    const d0=600, unit=(d0-z)/d0*k, widthPx=Math.ceil(length/unit), x0=xMin;
    const big=canvasOf(widthPx,heightPx), tiles=[];
    for(let sx=0;sx<widthPx;sx+=1024){
      const w=Math.min(1024,widthPx-sx), cv=canvasOf(w,heightPx), mesh=billboard(cv,w*unit,heightPx*unit);
      mesh.position.set(x0+(sx+w/2)*unit,0,z);scene.add(mesh);tiles.push({cv,sx,mesh});
    }
    const redraw=()=>{draw(big.getContext('2d'),widthPx,heightPx,x0,unit);for(const t of tiles){const ctx=t.cv.getContext('2d');ctx.clearRect(0,0,t.cv.width,heightPx);ctx.drawImage(big,-t.sx,0);t.mesh.material.map.needsUpdate=true;}};
    redrawers.push(redraw);
  }
  strip(-1700,1.25,120,drawFar);
  strip(-620,1.1,84,drawMid);

  // Ground: grass behind, the sidewalk Diego walks on, the road, and grass in front.
  function groundStrip(z0,z1,cv,repeatW,drawFn,plainColor) {
    const depth=z1-z0, geo=new THREE.PlaneGeometry(length,depth);geo.rotateX(-Math.PI/2);
    let mat;
    if(cv){const tx=texture(cv);tx.wrapS=THREE.RepeatWrapping;tx.wrapT=THREE.RepeatWrapping;tx.repeat.set(length/repeatW,cv===grassCv?depth/repeatW:1);mat=new THREE.MeshBasicMaterial({map:tx});redrawers.push(()=>{drawFn(cv.getContext('2d'));tx.needsUpdate=true;});}
    else {mat=new THREE.MeshBasicMaterial({color:0xffffff});redrawers.push(()=>mat.color.set(p[plainColor]));}
    const mesh=new THREE.Mesh(geo,mat);mesh.position.set(xMin+length/2,0,(z0+z1)/2);scene.add(mesh);return mesh;
  }
  const walkCv=canvasOf(32,28), roadCv=canvasOf(32,46), grassCv=canvasOf(64,64);
  groundStrip(-2600,0,null,0,null,'grass');
  groundStrip(0,28,walkCv,32,drawWalk);
  groundStrip(28,74,roadCv,32,drawRoad);
  groundStrip(74,7000,grassCv,64,drawGrass);

  // Trees and poplars behind the rooms; tufts in the foreground.
  const treeKinds=['round','poplar','bush'], treeCvs=treeKinds.map(k=>canvasOf(k==='bush'?36:40,k==='poplar'?52:48));
  const treeMats=[];
  treeCvs.forEach((cv,i)=>redrawers.push(()=>drawTree(cv.getContext('2d'),treeKinds[i])));
  const rt=rand(23);
  for(let x=xMin+40;x<xMax;x+=60+rt()*70){
    const kind=Math.floor(rt()*3), z=-150-rt()*260, s=(600-z)/600*(.8+rt()*.35), cv=treeCvs[kind];
    const mesh=new THREE.Mesh(new THREE.PlaneGeometry(cv.width*s,cv.height*s).translate(0,cv.height*s/2,0),treeMats[kind]||(treeMats[kind]=new THREE.MeshBasicMaterial({map:texture(cv),alphaTest:.5,side:THREE.DoubleSide})));
    mesh.position.set(x,0,z);scene.add(mesh);
  }
  const tuftCvs=[0,1,2].map(()=>canvasOf(16,12)), tuftMats=tuftCvs.map(cv=>new THREE.MeshBasicMaterial({map:texture(cv),alphaTest:.5,side:THREE.DoubleSide}));
  tuftCvs.forEach((cv,i)=>redrawers.push(()=>drawTuft(cv.getContext('2d'),i)));
  for(let x=xMin;x<xMax;x+=26+rt()*40){
    const i=Math.floor(rt()*3), z=90+rt()*240, s=.8+rt()*.5;
    const mesh=new THREE.Mesh(new THREE.PlaneGeometry(16*s,12*s).translate(0,6*s,0),tuftMats[i]);mesh.position.set(x,0,z);scene.add(mesh);
  }

  // Lamp posts between chapters carry the year of the room ahead.
  for(let i=0;i<N-1;i++){
    const label=(chapterEls[i+1].dataset.hud||'').replace('★','*');
    const cv=canvasOf(40,64), mesh=billboard(cv,40,64);mesh.position.set((i+.5)*GAP+10,0,30);scene.add(mesh);
    redrawers.push(()=>{drawLamp(cv.getContext('2d'),/^\d+$/.test(label)?label:'KM '+(i+1));mesh.material.map.needsUpdate=true;});
  }

  // A red Santiago bus keeps the street alive.
  const busCv=canvasOf(64,24), bus=billboard(busCv,64,24);bus.position.z=52;scene.add(bus);
  redrawers.push(()=>{drawBus(busCv.getContext('2d'));bus.material.map.needsUpdate=true;});

  // Every so often Diego runs or rides past, usually (80%) together with her.
  const sport={
    run:{cv:canvasOf(RUN_W,RUN_H),z:22,speed:78},
    bike:{cv:canvasOf(BIKE_W,BIKE_H),z:36,speed:140}
  };
  for(const k in sport){const s=sport[k];s.mesh=billboard(s.cv,s.cv.width,s.cv.height);s.mesh.position.z=s.z;s.mesh.visible=false;scene.add(s.mesh);}
  let outing=null, nextOuting=4+Math.random()*6;
  const SPORT_SPAN=720;
  redrawers.push(()=>{if(outing)drawSport(outing.s.cv.getContext('2d'),outing.kind,outing.together,clock);});
  function stepOuting(camX) {
    if(!outing&&clock>=nextOuting){
      const kind=Math.random()<.5?'run':'bike';
      outing={kind,s:sport[kind],together:Math.random()<.8,dir:Math.random()<.5?1:-1,start:clock};
      outing.s.mesh.visible=true;outing.s.mesh.scale.x=outing.dir;
    }
    if(!outing)return;
    const travelled=(clock-outing.start)*outing.s.speed;
    if(travelled>SPORT_SPAN*2){outing.s.mesh.visible=false;outing=null;nextOuting=clock+12+Math.random()*14;return;}
    outing.s.mesh.position.x=camX-outing.dir*SPORT_SPAN+outing.dir*travelled;
    drawSport(outing.s.cv.getContext('2d'),outing.kind,outing.together,clock);
    outing.s.mesh.material.map.needsUpdate=true;
  }

  // The rooms: pop-up pages hinged on the floor line.
  const rooms=keys.map((key,i)=>{
    const def=SCENES[key], cv=canvasOf(SW,SH), base=canvasOf(SW,SH), mesh=billboard(cv,SW,SH);
    mesh.position.set(i*GAP,0,0);scene.add(mesh);
    return {def,cv,base,mesh,i,last:-1,up:1,wasUp:false};
  });
  function paintRoom(room,t) {
    const ctx=room.cv.getContext('2d');ctx.clearRect(0,0,SW,SH);ctx.drawImage(room.base,0,0);
    c=ctx;room.def.anim(t);room.mesh.material.map.needsUpdate=true;
  }
  redrawers.push(()=>rooms.forEach(room=>{const ctx=room.base.getContext('2d');ctx.clearRect(0,0,SW,SH);c=ctx;room.def.base();paintRoom(room,clock);}));

  // Diego.
  const diegoCv=canvasOf(DW,DH), diego=billboard(diegoCv,DW,DH);scene.add(diego);

  // Pixel confetti for pop-ups and outfit swaps (square points are already pixels).
  const MAX=360, pPos=new Float32Array(MAX*3), pCol=new Float32Array(MAX*3), parts=[];
  const pGeo=new THREE.BufferGeometry();pGeo.setAttribute('position',new THREE.BufferAttribute(pPos,3));pGeo.setAttribute('color',new THREE.BufferAttribute(pCol,3));
  const points=new THREE.Points(pGeo,new THREE.PointsMaterial({size:2.6,vertexColors:true,sizeAttenuation:true}));points.frustumCulled=false;scene.add(points);
  const tmpColor=new THREE.Color();
  function burst(x,y,z,n,spread,palette) {
    if(motion.matches)return;
    for(let k=0;k<n;k++){if(parts.length>=MAX)parts.shift();parts.push({x:x+(Math.random()-.5)*spread,y:y+Math.random()*4,z:z+Math.random()*8,vx:(Math.random()-.5)*60,vy:40+Math.random()*80,vz:Math.random()*30,life:.9+Math.random()*.6,color:palette[k%palette.length]});}
  }
  function stepParticles(dt) {
    for(let k=parts.length-1;k>=0;k--){const q=parts[k];q.life-=dt;if(q.life<=0||q.y<0){parts.splice(k,1);continue;}q.vy-=180*dt;q.x+=q.vx*dt;q.y+=q.vy*dt;q.z+=q.vz*dt;}
    for(let k=0;k<MAX;k++){const q=parts[k];if(q){pPos[k*3]=q.x;pPos[k*3+1]=q.y;pPos[k*3+2]=q.z;tmpColor.set(q.color);pCol[k*3]=tmpColor.r;pCol[k*3+1]=tmpColor.g;pCol[k*3+2]=tmpColor.b;}else{pPos[k*3+1]=-999;}}
    pGeo.attributes.position.needsUpdate=true;pGeo.attributes.color.needsUpdate=true;
  }

  // HUD and chapter navigation.
  const yearEl=document.getElementById('tj-year'), hudLabel=document.getElementById('tj-hud-label'), hudCount=document.getElementById('tj-hud-count'), hint=document.getElementById('tj-hint');
  const progress=document.getElementById('tj-progress');
  const navLinks=chapterEls.map((el,i)=>{
    const a=document.createElement('a');a.href=`#${el.id}`;a.innerHTML=`<i></i><span>${String(i).padStart(2,'0')} · ${el.dataset.label}</span>`;a.setAttribute('aria-label',`${String(i).padStart(2,'0')}: ${el.dataset.label}`);
    // Land with the card on the anchor line, so Diego is standing in the room, not mid-street.
    a.addEventListener('click',event=>{event.preventDefault();scrollTo({top:anchors[i]-vh*anchorRatio,behavior:motion.matches?'auto':'smooth'});history.replaceState(null,'',`#${el.id}`);cards[i].focus({preventScroll:true});});
    progress.append(a);return a;
  });
  cards.forEach(card=>card.setAttribute('tabindex','-1'));

  /* Theme: follow the page switch (data-theme) and the home's dusk setting. */
  function readMode() { if(document.documentElement.dataset.theme==='night')return 'night'; try{if(localStorage.getItem('diegulio-time')==='dusk')return 'dusk';}catch{} return 'day'; }
  function applyTheme() {
    mode=readMode();p=palettes[mode];
    scene.background=new THREE.Color(p.haze);scene.fog.color.set(p.haze);
    redrawers.forEach(fn=>fn());
    textures.forEach(tx=>{tx.needsUpdate=true;});
    resizeSky();lastSky=-1;
  }
  new MutationObserver(applyTheme).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});

  /* Layout */
  let vw=1, vh=1, anchors=[], anchorRatio=.5, headerH=88;
  function resizeSky() {
    // Sky pixels match the scene's: about 2.5 CSS pixels each.
    const aspect=vw/vh, h=Math.max(120,Math.min(560,Math.round(vh*1.6/2.5))), w=Math.max(16,Math.round(h*aspect));
    if(skyCanvas.width!==w){skyCanvas.width=w;skyCanvas.height=h;sky.material.map.dispose();sky.material.map=texture(skyCanvas);}
    const dist=8000, hh=2*dist*Math.tan(THREE.MathUtils.degToRad(camera.fov/2));
    sky.scale.set(hh*aspect*1.6,hh*1.6,1);
  }
  function measure() {
    vw=stage.clientWidth;vh=stage.clientHeight;headerH=document.querySelector('.site-header')?.offsetHeight||0;
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(vw,vh,false);camera.aspect=vw/vh;camera.updateProjectionMatrix();
    anchorRatio=vw<=800?.36:.5;
    anchors=cards.map(card=>{const r=card.getBoundingClientRect();return r.top+scrollY+r.height/2;});
    resizeSky();lastSky=-1;
  }
  function scrollIndex() {
    const line=scrollY+vh*anchorRatio;
    if(line<=anchors[0])return 0;
    for(let i=0;i<N-1;i++)if(line<anchors[i+1])return i+(line-anchors[i])/Math.max(1,anchors[i+1]-anchors[i]);
    return N-1;
  }

  /* Camera framing: the room sits in the free area beside (desktop) or above (mobile) the card. */
  const fovTan=Math.tan(THREE.MathUtils.degToRad(15));
  function frameFor(bump) {
    // Desktop: the room sits right of the card. Mobile: the card floats on the sky, the room below it.
    const desktop=vw>800;
    const cardRight=desktop?Math.min(vw*.48,(cards[0].getBoundingClientRect().right||460)+24):0;
    const fx0=cardRight, fx1=vw-(desktop?32:6), fy0=desktop?headerH+24:vh*.52, fy1=desktop?vh-76:vh-64;
    const fw=Math.max(200,fx1-fx0), fh=Math.max(140,fy1-fy0);
    const wpp=Math.max(SW*(desktop?1.1:1.04)/fw,(SH+34)/(fh*(desktop?.62:.85)));
    const centerX=lerp((fx0+fx1)/2,vw/2,bump*.6), centerY=lerp((fy0+fy1)/2,vh*.5,bump*.4);
    return {dist:wpp*vh/(2*fovTan), cx:centerX, cy:centerY};
  }

  /* Animation state */
  let fs=0, clock=0, previous=performance.now(), running=false, frameId=0, lastSky=-1;
  let charX=rooms[0].i*GAP-SW/2+rooms[0].def.spot, facing=1, stepDist=0, lastMove=0, outfit=rooms[0].def.shirt, activeIndex=-1, lastHud='';
  const blinkAt=()=>Math.floor(clock*10)%37===0;

  function update(dt) {
    const reduced=motion.matches;
    const raw=scrollIndex(), target=Number.isFinite(raw)?Math.min(N-1,Math.max(0,raw)):0;
    fs=reduced?target:fs+(target-fs)*(1-Math.exp(-dt*7));
    if(Math.abs(target-fs)<.0005)fs=target;
    const i=Math.max(0,Math.min(N-2,Math.floor(fs))), u=Math.min(1,fs-i);
    const e=smooth(.2,.8,u), walk=smooth(.1,.9,u), g=i+e;
    const bump=reduced?0:Math.sin(Math.PI*e);

    // Camera dolly: pull back and rise while travelling, settle on arrival.
    const fr=frameFor(bump);
    const dist=fr.dist*(1+.32*bump), elev=THREE.MathUtils.degToRad(7+10*bump);
    const camX=lerp(i*GAP,(i+1)*GAP,e), aimY=SH*.48-bump*10;
    camera.position.set(camX,aimY+Math.sin(elev)*dist,Math.cos(elev)*dist);
    camera.lookAt(camX,aimY,0);
    camera.setViewOffset(vw,vh,-(fr.cx-vw/2),-(fr.cy-vh/2),vw,vh);

    // Pages fold down behind Diego and pop up ahead of him.
    for(const room of rooms){
      const d=Math.abs(g-room.i), up=1-smooth(.15,.55,d);
      const shaped=reduced?up:backOut(up);
      room.mesh.rotation.x=-(1-shaped)*Math.PI/2;
      room.mesh.material.color.setScalar(.62+.38*Math.max(0,Math.min(1,shaped)));
      room.mesh.visible=d<2.2;
      if(up>.9&&!room.wasUp&&d<.6){burst(room.i*GAP,2,4,46,SW*.9,[colors.lime,colors.cream,colors.gold,colors.rust]);}
      room.wasUp=up>.9;
      if(room.mesh.visible&&d<1.3&&Math.floor(clock*12)!==room.last){room.last=Math.floor(clock*12);paintRoom(room,reduced?0:clock);}
    }

    // Diego walks the sidewalk between rooms and steps into each one.
    const from=rooms[i], to=rooms[i+1];
    const spotA=from.i*GAP-SW/2+from.def.spot, spotB=to.i*GAP-SW/2+to.def.spot;
    const nx=lerp(spotA,spotB,walk), dx=nx-charX;charX=nx;
    const outside=smooth(0,.1,Math.min(walk,1-walk));
    diego.position.set(charX,4*(1-outside),1+15*outside);
    if(Math.abs(dx)>.01){stepDist+=Math.abs(dx);lastMove=clock;facing=Math.sign(dx);}
    const travelling=walk>.002&&walk<.998, moving=clock-lastMove<.15;
    const here=walk<.5?from:to, nextOutfit=here.def.shirt;
    if(nextOutfit!==outfit){outfit=nextOutfit;burst(charX,10,18,18,10,[colors.lime,colors.cream]);}
    const pose=travelling?(moving?'walk':'stand'):here.def.pose;
    if(!travelling)facing=here.def.facing;
    diego.scale.x=facing;
    c=diegoCv.getContext('2d');c.clearRect(0,0,DW,DH);
    poses[pose](20,DH-1,reduced?0:clock,{shirt:outfit,step:Math.floor(stepDist/5)%4,blink:blinkAt()});
    diego.material.map.needsUpdate=true;

    // Street life and sky.
    bus.position.x=camX+900-((clock*34)%1800);
    stepOuting(camX);
    stepParticles(dt);
    const skyTick=Math.floor(clock*6);
    if(skyTick!==lastSky){lastSky=skyTick;drawSky(skyCtx,skyCanvas.width,skyCanvas.height,reduced?0:clock);sky.material.map.needsUpdate=true;}

    // HUD: count the years while walking, highlight the chapter in view.
    const nearest=Math.round(g);
    const a=chapterEls[i].dataset.hud, b=chapterEls[i+1].dataset.hud;
    const hud=(/^\d+$/.test(a)&&/^\d+$/.test(b))?String(Math.round(lerp(+a,+b,e))):(e<.5?a:b);
    if(hud!==lastHud){yearEl.textContent=hud;lastHud=hud;if(!reduced){yearEl.classList.remove('is-flipping');void yearEl.offsetWidth;yearEl.classList.add('is-flipping');}}
    if(nearest!==activeIndex){
      activeIndex=nearest;
      hudLabel.textContent=chapterEls[nearest].dataset.label.toUpperCase();
      hudCount.textContent=`${String(nearest).padStart(2,'0')} / ${String(N-1).padStart(2,'0')}`;
      cards.forEach((card,k)=>card.classList.toggle('is-active',k===nearest));
      navLinks.forEach((link,k)=>{if(k===nearest)link.setAttribute('aria-current','step');else link.removeAttribute('aria-current');});
    }
    hint.classList.toggle('is-hidden',fs>.15);
  }

  function tick(now) {
    const dt=Math.min(.05,(now-previous)/1000);previous=now;
    clock+=motion.matches?0:dt;
    update(dt);
    renderer.render(scene,camera);
    frameId=running?requestAnimationFrame(tick):0;
  }
  function play() { if(running)return; running=true; previous=performance.now(); frameId=requestAnimationFrame(tick); }
  function pause() { running=false; cancelAnimationFrame(frameId); frameId=0; }

  journey.classList.add('tj-ready');
  measure();applyTheme();
  fs=scrollIndex();
  addEventListener('resize',measure);
  new ResizeObserver(measure).observe(journey);
  new IntersectionObserver(entries=>{entries[0].isIntersecting&&!document.hidden?play():pause();}).observe(stage);
  document.addEventListener('visibilitychange',()=>{document.hidden?pause():play();});
  play();
}
