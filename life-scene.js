/* Diego's little world. Everything is drawn on a 640 × 416 pixel grid.
   No generated video, image downloads, WebGL or third-party runtime needed. */
const canvas = document.getElementById('life-canvas');
const W = 640, H = 416;
const display = canvas.getContext('2d');
const raster = document.createElement('canvas');
raster.width = W; raster.height = H;
const ctx = raster.getContext('2d');
const backdrop = document.createElement('canvas');
backdrop.width = W; backdrop.height = H;
let c = ctx;
let timeOfDay = 'day';
try { const saved = localStorage.getItem('diegulio-time'); if (['day','dusk','night'].includes(saved)) timeOfDay = saved; } catch {}
let dusk = timeOfDay === 'dusk', night = timeOfDay === 'night', zone = 'all', elapsed = 0;
const motion = matchMedia('(prefers-reduced-motion: reduce)');
let paused = motion.matches, visible = true, frame = 0, previous = 0;
let cameraMoving = false;
const camera = { x: W / 2, y: H / 2, zoom: 1 };
let cameraTarget = { ...camera };

const colors = {
  ink:'#33463e', dark:'#253d35', cream:'#f3eed9', paper:'#e6e4d0',
  green:'#8caa5e', leaf:'#64865a', lime:'#c9df85', rust:'#bc7352',
  skin:'#bf8c63', skinLight:'#dbac7f', hair:'#293631', white:'#f9f5df',
  blue:'#537c8e', navy:'#3b5870', terracotta:'#c78968', wood:'#ae9067'
};
let p;
function palette() {
  p = night ? {sky:'#172839', skyLow:'#263c4d', mountain:'#3b5366', mountainDark:'#2d4353', snow:'#8da6b3', hill:'#344f48', hillLight:'#4c6655', city:'#425958', wall:'#a19c80', room:'#d4c59c', roomShade:'#a7ac82', water:'#356a7b', waterLight:'#7cafb4', road:'#304347'}
    : dusk ? {sky:'#c4aba9', skyLow:'#ead1b3', mountain:'#888c9c', mountainDark:'#717c8c', snow:'#eaded5', hill:'#748671', hillLight:'#93a080', city:'#aba999', wall:'#d9cdb7', room:'#ded4bd', roomShade:'#c9c7ac', water:'#6c9da6', waterLight:'#acd0c6', road:'#555f5b'}
    : {sky:'#acd0d2', skyLow:'#d4e2d3', mountain:'#94acb5', mountainDark:'#78959e', snow:'#f2f0de', hill:'#8ca886', hillLight:'#acbd89', city:'#bac6b3', wall:'#e7e2ce', room:'#f0ecd9', roomShade:'#dce0c6', water:'#74b4bc', waterLight:'#c3e4d6', road:'#58655e'};
}
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
function circle(x,y,r,color) {
  for(let dy=-r;dy<=r;dy++){const dx=Math.floor(Math.sqrt(r*r-dy*dy));R(x-dx,y+dy,dx*2+1,1,color);}
}
function outline(x,y,w,h,color) {R(x,y,w,1,color);R(x,y+h-1,w,1,color);R(x,y,1,h,color);R(x+w-1,y,1,h,color);}

// A true 3 × 5 bitmap alphabet keeps every sign on the same pixel grid.
const glyphs = {
 A:'010101111101101',B:'110101110101110',C:'011100100100011',D:'110101101101110',E:'111100110100111',F:'111100110100100',G:'011100101101011',H:'101101111101101',I:'111010010010111',J:'001001001101010',K:'101101110101101',L:'100100100100111',M:'101111111101101',N:'101111111111101',O:'010101101101010',P:'110101110100100',Q:'010101101111011',R:'110101110101101',S:'011100010001110',T:'111010010010010',U:'101101101101111',V:'101101101101010',W:'101101111111101',X:'101101010101101',Y:'101101010010010',Z:'111001010100111',
 '0':'111101101101111','1':'010110010010111','2':'110001010100111','3':'110001010001110','4':'101101111001001','5':'111100110001110','6':'011100111101111','7':'111001010010010','8':'111101111101111','9':'111101111001110',
 ' ':'000000000000000','.':'000000000000010',':':'000010000010000','/':'001001010100100','-':'000000111000000','+':'000010111010000','=':'000111000111000','>':'100010001010100','<':'001010100010001','!':'010010010000010','?':'110001010000010','&':'010101010101011','(':'010100100100010',')':'010001001001010','_':'000000000000111'
};
function text(value,x,y,color=colors.ink,scale=1) {
  [...value.toUpperCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'')].forEach((ch,index)=>{
    const bits=glyphs[ch]||glyphs[' '];
    for(let k=0;k<15;k++)if(bits[k]==='1')R(x+index*4*scale+(k%3)*scale,y+Math.floor(k/3)*scale,scale,scale,color);
  });
}
function sign(label,x,y,w,color=colors.ink,ink=colors.cream) { R(x,y,w,12,color);text(label,x+Math.floor((w-label.length*4+1)/2),y+4,ink); }
function plant(x,y,size=1) { R(x-3,y,7,5,colors.rust);R(x-1,y-9*size,2,9*size,colors.leaf);R(x-6,y-7*size,5,4*size,colors.leaf);R(x+1,y-10*size,5,5*size,colors.green); }
function tree(x,y,size=1) {
  R(x-2,y-18*size,4,20*size,colors.wood);
  R(x-12*size,y-36*size,24*size,16*size,colors.leaf);
  R(x-17*size,y-29*size,34*size,10*size,colors.leaf);
  R(x-7*size,y-41*size,15*size,12*size,colors.green);
  R(x-13*size,y-29*size,14*size,6*size,colors.green);
}
function monitor(x,y,w=27,h=18) { R(x,y,w,h,colors.ink);R(x+2,y+2,w-4,h-5,'#2d4840');R(x+w/2-1,y+h,3,4,colors.ink);R(x+w/2-6,y+h+4,13,2,colors.ink); }
function desk(x,y,w) { R(x,y,w,3,colors.wood);R(x+3,y+3,3,13,colors.wood);R(x+w-5,y+3,3,13,colors.wood); }
function books(x,y,n=7) { for(let i=0;i<n;i++){const h=6+(i*7%6);R(x+i*4,y-h,3,h,[colors.rust,colors.blue,colors.green,colors.wood][i%4]);R(x+i*4,y-3,3,1,colors.cream);} }
function windowBox(x,y,w,h) {R(x,y,w,h,colors.wood);R(x+2,y+2,w-4,h-4,dusk||night?'#f6d59b':'#b9d8cd');R(x+w/2,y+2,1,h-4,colors.wood);}

function drawLandscape() {
  // Snow-covered Andes. Pixel silhouettes are intentionally illustrative.
  poly([[0,154],[0,110],[41,83],[67,109],[113,40],[143,77],[173,17],[202,63],[222,54],[267,117],[314,58],[350,89],[394,30],[423,70],[443,57],[485,99],[544,23],[577,68],[610,58],[640,100],[640,192]],p.mountain);
  poly([[94,68],[113,40],[143,77],[127,69],[117,57],[110,64],[104,62]],p.snow);
  poly([[148,53],[173,17],[202,63],[182,50],[174,33],[165,45],[160,41]],p.snow);
  poly([[372,59],[394,30],[418,65],[400,55],[392,44],[385,56]],p.snow);
  poly([[520,55],[544,23],[575,66],[554,52],[543,37],[536,51],[530,46]],p.snow);
  poly([[0,150],[76,116],[100,133],[166,76],[224,127],[258,116],[307,150],[379,92],[440,122],[492,87],[554,127],[602,101],[640,134],[640,202],[0,202]],p.mountainDark);
  // Santiago skyline and a stylized Costanera tower.
  for(let i=0;i<27;i++){const x=i*26-8,h=9+(i*13%28);R(x,198-h,18,h,p.city);for(let wy=202-h;wy<194;wy+=6)R(x+3,wy,11,2,'#d4d7c2');}
  poly([[584,182],[587,100],[592,93],[597,100],[600,182]],'#bdc9be');
  for(let y=111;y<180;y+=5)R(588,y,9,1,'#8fa8a1');
  R(592,83,1,11,p.city);
  // Cerro San Cristóbal, statue and walking trail.
  poly([[219,238],[253,194],[282,157],[304,146],[328,119],[347,107],[362,116],[382,144],[411,168],[447,203],[453,254]],p.hill);
  poly([[230,238],[298,174],[321,164],[347,126],[358,126],[334,172],[350,194],[304,211],[293,253]],p.hillLight);
  line(280,246,309,215,'#c3c39a',3);line(309,215,346,195,'#c3c39a',3);line(346,195,326,170,'#c3c39a',3);line(326,170,350,136,'#c3c39a',2);
  R(337,108,23,5,'#a9ad96');R(342,98,12,10,'#d4d6bf');
  R(346,83,5,15,colors.cream);R(344,91,9,9,colors.cream);R(347,79,3,4,colors.cream);R(348,75,1,4,colors.cream);
  text('SAN CRISTOBAL',314,125,'#536e55');
  for(let i=0;i<9;i++){const x=269+(i*31%134),y=172+(i*13%33);R(x,y,5,4,'#75966f');R(x+1,y-3,3,4,'#75966f');}
  line(266,142,431,91,'#75867a');R(269,137,2,32,'#829286');R(426,89,2,52,'#829286');
  R(0,306,W,60,'#b0bf8b');
  for(let i=0;i<95;i++){const x=(i*71)%640,y=310+(i*19)%58;R(x,y,2,1,i%2?'#95ad77':'#c5cf9c');}
}

function drawBuildings() {
  // Open-front three-storey laboratory: professional work, teaching, own projects.
  R(27,148,236,172,'#829483');R(31,152,228,161,p.wall);
  for(const y of [154,207,260]) {R(35,y+3,220,46,p.room);R(35,y+36,220,13,p.roomShade);R(29,y+49,233,5,'#9caa94');R(30,y+53,232,2,'#c4c9ac');}
  for(const x of [30,255])R(x,151,5,166,'#c1c5af');
  R(24,144,242,5,colors.cream);R(24,149,242,3,'#788e7b');
  R(35,125,2,20,colors.ink);R(129,125,2,20,colors.ink);
  sign('DIEGULIO LAB',29,111,112,colors.ink,colors.lime);R(30,123,111,2,colors.wood);
  plant(170,139);plant(194,139);plant(221,139);
  R(153,139,80,5,colors.wood);R(159,128,72,1,'#9aa893');R(157,128,1,11,'#9aa893');R(231,128,1,11,'#9aa893');
  // Data science floor.
  text('01 / DATA SCIENCE',40,160,'#7c8974');sign('VISA',213,158,35,'#506c80');
  monitor(47,175,34,18);desk(41,198,65);monitor(128,173,49,23);desk(122,198,66);
  text('MODEL > INSIGHT',130,166,'#8c987d');
  R(201,178,41,19,'#e1debe');for(let i=0;i<6;i++)R(205+i*5,193-i*2,3,3+i*2,colors.green);
  R(203,197,45,2,colors.wood);plant(239,194,.65);
  // Applied ML classroom.
  text('02 / USACH',40,213,'#7c8974');
  R(42,224,89,29,colors.wood);R(44,226,85,25,'#466853');
  text('APPLIED ML',50,230,'#d7e5ae');text('Y = F(X)',51,241,colors.cream);
  for(let i=0;i<3;i++){desk(169+i*25,246,20);R(172+i*25,239,12,7,colors.navy);R(173+i*25,240,10,4,'#bed2bf');}
  // Code & coffee studio.
  text('03 / CODIGO + CAFE',40,266,'#7c8974');
  windowBox(44,278,34,27);desk(92,302,82);monitor(122,281,36,17);
  R(109,297,5,5,colors.cream);R(113,298,3,2,colors.cream);
  R(181,278,35,29,colors.wood);R(184,280,29,10,'#d2d2b8');R(184,293,29,11,'#d2d2b8');books(185,290,6);books(185,304,6);
  R(227,278,21,29,colors.dark);for(let y=281;y<305;y+=5){R(230,y,14,2,'#617968');R(242,y,2,1,colors.lime);}
  plant(83,304,.85);

  // Training club / earlier work: another dollhouse building.
  R(426,149,188,115,'#829483');R(430,152,180,108,p.wall);
  R(433,157,173,45,p.room);R(433,195,173,8,p.roomShade);
  R(431,204,180,5,'#a7af96');R(433,211,173,43,p.room);R(433,247,173,7,p.roomShade);
  R(424,145,191,5,colors.cream);R(437,127,2,18,colors.ink);R(541,127,2,18,colors.ink);
  sign('FUERZA + CONSTANCIA',432,116,124,colors.rust);
  plant(588,140);R(572,134,1,11,colors.wood);R(601,134,1,11,colors.wood);R(572,134,30,1,colors.wood);
  text('GYM / UNA REP MAS',439,159,'#7c8974');
  // Squat rack and dumbbells.
  R(443,172,3,28,colors.ink);R(469,172,3,28,colors.ink);R(443,172,29,2,colors.ink);R(439,199,36,2,colors.ink);
  R(484,192,32,4,colors.wood);R(488,196,3,5,colors.ink);R(510,196,3,5,colors.ink);
  for(let x=490;x<516;x+=9){R(x,184,2,8,colors.ink);R(x-2,185,6,2,colors.ink);}
  // Treadmill.
  R(548,197,39,3,colors.ink);R(581,178,3,20,colors.ink);R(574,176,15,3,colors.ink);R(578,173,7,4,colors.blue);plant(600,198,.65);
  text('ANTES DE ESTE COMMIT...',439,216,'#7c8974');
  sign('FALABELLA 2022',439,227,73,'#839c59');sign('WALMART 2023',520,227,79,'#537f96');
  desk(442,250,59);monitor(449,240,22,9);desk(519,250,74);monitor(538,240,23,9);plant(597,247,.65);

  // Outdoor reading bench, bike rack and a tiny Chilean flag.
  R(281,283,36,3,colors.wood);R(281,276,36,5,colors.wood);R(285,286,2,6,colors.ink);R(313,286,2,6,colors.ink);
  tree(289,272,.75);tree(408,252,.65);
  R(279,191,2,37,colors.wood);R(274,225,12,3,'#a5ad8a');
  // Pool with three swimming lanes.
  R(330,270,278,54,'#e9e5cc');R(335,275,268,43,'#528d9a');R(338,278,262,37,p.water);
  R(332,324,275,4,'#90a489');
  for(const y of [290,303])for(let x=339;x<599;x+=8)R(x,y,5,1,(x+y)%3===0?'#d99168':colors.cream);
  R(341,283,3,2,'#ddded0');R(598,278,2,37,'#aed7cf');
  sign('SWIM',556,258,42,'#537e84');
  for(let i=0;i<3;i++){R(348+i*72,266,17,3,'#a5af97');R(353+i*72,259,8,7,colors.cream);R(350+i*72,258,14,3,colors.blue);}
  for(const x of [591,597]){R(x,277,1,16,colors.cream);R(x-2,275,3,2,colors.cream);}R(591,285,7,1,colors.cream);R(591,290,7,1,colors.cream);
  // Front path and bike lane; a subtle layer of Chilean street life.
  R(0,330,W,3,'#e4ddbe');R(0,333,W,28,colors.terracotta);R(0,340,W,1,'#dfb28b');R(0,359,W,2,'#e3bd98');
  R(0,363,W,10,'#bccaa1');R(0,373,W,3,'#e8dfbd');R(0,376,W,29,p.road);R(0,405,W,3,colors.cream);
  for(let x=2;x<W;x+=31)R(x,389,15,1,'#d5d9b1');
  R(0,408,W,8,'#a6b88d');
  for(let x=14;x<640;x+=47){R(x,369,4,3,'#90a875');R(x+2,366,1,3,'#90a875');}
  sign('RUN',24,322,24,colors.leaf);R(36,334,1,7,colors.ink);
  sign('BIKE',569,366,28,colors.blue);
  // Street lamps.
  for(const x of [13,321,626]){R(x,289,2,40,colors.ink);R(x-3,287,8,2,colors.ink);R(x-2,282,6,5,dusk||night?'#f4d48a':'#e6e6ca');R(x-3,281,8,1,colors.ink);}
}

function bake() {
  palette();c=backdrop.getContext('2d');c.clearRect(0,0,W,H);drawLandscape();
  if(night) {
    // Shade only the landscape pixels; keep the starry sky transparent.
    c.globalCompositeOperation='source-atop';R(0,0,W,H,'#10253d40');c.globalCompositeOperation='source-over';
    for(let i=0;i<27;i++){const x=i*26-8,h=9+(i*13%28);for(let y=202-h;y<194;y+=6)if((i+y)%3)R(x+3,y,3,2,'#d9bd7b');}
    for(let y=113;y<180;y+=10)R(589,y,2,2,'#d9bd7b');
  }
  drawBuildings();
  if(night) {
    // Exterior paths stay subdued while the open rooms remain warmly lit.
    R(0,330,W,86,'#10253d50');
    for(const x of [13,321,626]){poly([[x-1,288],[x+3,288],[x+20,337],[x-18,337]],'#f8d18422');R(x-15,337,32,2,'#f8d18435');}
    for(const y of [282,298,312]){R(598,y,2,2,'#fbe4a5');R(588,y+1,9,1,'#e4cf8540');}
  }
  c=ctx;
}

// The same small sprite appears in different outfits, poses and places.
function head(x,y,glasses=false,cap=false) {
  R(x+1,y,5,2,colors.hair);R(x,y+2,7,3,colors.hair);
  R(x+1,y+3,6,5,colors.skin);R(x+4,y+3,3,4,colors.skinLight);R(x+6,y+5,2,2,colors.skinLight);
  R(x+1,y+4,1,2,colors.hair);R(x+5,y+4,1,1,colors.ink);
  if(glasses){R(x+3,y+3,5,2,colors.dark);R(x+4,y+3,1,1,'#738e85');}
  R(x+3,y+8,3,2,colors.skin);
  if(cap){R(x,y,7,2,colors.lime);R(x+5,y+1,4,1,colors.lime);}
}
function seated(x,y,t,shirt=colors.blue) {
  head(x,y);R(x,y+9,8,7,shirt);R(x-2,y+10,2,11,colors.ink);R(x-3,y+20,13,2,colors.ink);
  R(x+7,y+11,4,2,shirt);R(x+10,y+12,5,2,colors.skin);R(x+14,y+11+(Math.floor(t*5)%2),3,2,colors.skinLight);
  R(x+2,y+16,8,3,colors.ink);R(x+8,y+17,3,6,colors.ink);R(x+7,y+22,6,2,colors.white);R(x,y+22,3,4,colors.ink);
}
function running(x,y,t,shirt=colors.lime,scale=1) {
  c.save();c.translate(Math.round(x),Math.round(y));c.scale(scale,scale);
  const step=Math.floor(t*7)%4, bob=step%2;
  head(0,bob,true);
  R(0,9+bob,8,7,shirt);R(1,16+bob,7,3,colors.navy);
  const swing=[4,0,-4,0][step];
  line(2,19,2+swing,24,colors.skin,2);line(2+swing,24,5+swing,26,colors.skin,2);R(4+swing,26,5,2,colors.white);
  line(6,19,6-swing,23,colors.skin,2);line(6-swing,23,4-swing,26,colors.ink,2);R(3-swing,26,5,2,colors.white);
  line(1,10,-3,14+Math.max(0,swing/2),colors.skin,2);line(-3,14+Math.max(0,swing/2),-5+swing/2,11,colors.skin,2);
  line(7,10,11,14-Math.max(0,swing/2),colors.skin,2);line(11,14-Math.max(0,swing/2),13-swing/2,10,colors.skin,2);
  R(-3,13,2,2,colors.ink);c.restore();
}
function teaching(x,y,t) {
  head(x,y);R(x,y+9,8,10,colors.rust);R(x+1,y+19,3,8,colors.ink);R(x+5,y+19,3,8,colors.ink);R(x,y+27,4,2,colors.white);R(x+5,y+27,5,2,colors.white);
  const arm=Math.floor(t*.9)%2;line(x+1,y+10,x-6,y+8-arm*3,colors.skin,2);line(x-6,y+8-arm*3,x-20,y+3-arm*4,colors.wood);
  R(x+8,y+11,2,7,colors.skin);R(x+8,y+16,4,4,colors.cream);
}
function lifting(x,y,t) {
  head(x,y);R(x,y+9,8,9,colors.navy);R(x+1,y+18,3,7,colors.ink);R(x+5,y+18,3,7,colors.ink);R(x,y+25,4,2,colors.white);R(x+5,y+25,5,2,colors.white);
  const lift=Math.round((Math.sin(t*2)+1)*4);
  line(x,y+10,x-4,y+6-lift,colors.skin,2);line(x+7,y+10,x+11,y+6-lift,colors.skin,2);
  R(x-6,y+4-lift,21,2,colors.ink);R(x-8,y+1-lift,3,8,colors.ink);R(x+14,y+1-lift,3,8,colors.ink);R(x-10,y+2-lift,2,6,colors.ink);R(x+17,y+2-lift,2,6,colors.ink);
}
function bike(x,y,t) {
  c.save();c.translate(Math.round(x),Math.round(y));
  for(const wx of [0,25]){circle(wx,15,8,colors.ink);circle(wx,15,6,p.road);line(wx-5*Math.cos(t*9),15-5*Math.sin(t*9),wx+5*Math.cos(t*9),15+5*Math.sin(t*9),'#bac8ab');R(wx,15,2,1,colors.cream);}
  line(0,15,9,4,colors.rust,2);line(9,4,14,15,colors.rust,2);line(14,15,0,15,colors.rust);line(9,4,22,4,colors.rust);line(22,4,14,15,colors.rust);line(20,0,25,15,colors.cream);line(20,0,25,-1,colors.ink);R(6,1,7,2,colors.ink);
  head(13,-17,true,true);line(15,-7,9,1,colors.lime,5);line(16,-6,22,-3,colors.skin,2);line(22,-3,24,0,colors.skin,2);
  const a=t*8;line(10,1,14+Math.cos(a)*4,8+Math.sin(a)*3,colors.skin,3);line(14+Math.cos(a)*4,8+Math.sin(a)*3,14+Math.cos(a)*4,15+Math.sin(a)*4,colors.skin,2);R(13+Math.cos(a)*4,15+Math.sin(a)*4,5,2,colors.white);
  c.restore();
}
function swimmer(x,y,t) {
  const stroke=Math.floor(t*6)%4;
  R(x-13,y+1,16,4,colors.skin);R(x-17,y+2,7,3,colors.navy);R(x+1,y-1,6,5,colors.skinLight);R(x+1,y-2,6,2,colors.lime);R(x+5,y,3,1,colors.ink);
  const armY=[-5,-3,1,2][stroke];line(x-3,y+2,x-3,y+armY,colors.skin,2);line(x-3,y+armY,x+4,y+armY-1,colors.skinLight,2);
  for(let i=0;i<4;i++)R(x-23+i*8,y+5+(i+stroke)%2,5,1,p.waterLight);
  if(stroke<2){R(x-16,y-2,2,1,p.waterLight);R(x+8,y+1,2,1,colors.white);}
}

function drawSky(t) {
  R(0,0,W,H,p.sky);R(0,84,W,70,p.skyLow);
  if(night) {
    for(let i=0;i<65;i++) {
      const x=(i*97+23)%W,y=7+(i*43)%97;
      R(x,y,1,1,Math.sin(t*.65+i*2)>.4?'#e4e5bd':'#788e9c');
      if(i%13===0){R(x-1,y,3,1,'#a5bbc1');R(x,y-1,1,3,'#a5bbc1');}
    }
    circle(481,37,15,'#a6bdbe');circle(481,37,12,'#f4edbd');circle(487,31,12,p.sky);
    return;
  }
  circle(dusk?541:481,dusk?67:37,16,dusk?'#f1c17f':'#ece7b0');circle(dusk?541:481,dusk?67:37,11,dusk?'#ffe0a3':'#fff4c4');
  for(let i=0;i<5;i++){const x=((i*149+t*(1.8+i*.3))%(W+100))-50,y=29+(i*31%62);R(x,y,30,5,'#edf0df');R(x+6,y-4,19,5,'#edf0df');R(x+11,y-7,9,4,'#edf0df');}
  for(let i=0;i<4;i++){const x=310+i*12+Math.floor(t*2)%80,y=39+(i%2)*4;const f=Math.floor(t*3+i)%2;line(x,y,x+2,y-f-1,'#6c8a80');line(x+2,y-f-1,x+4,y,'#6c8a80');}
}
function drawLife(t) {
  // Cable cars drifting across the hill.
  for(let i=0;i<2;i++) {const progress=(t*.016+i*.53)%1,x=267+progress*163,y=142-progress*51;R(x,y,1,6,colors.ink);R(x-5,y+6,11,9,i?colors.cream:colors.rust);R(x-4,y+7,3,4,'#b7d3cd');R(x+1,y+7,3,4,'#b7d3cd');R(x-4,y+15,9,1,colors.ink);}
  // Chilean flag, three discrete waving frames.
  const flutter=Math.floor(t*3)%3;
  R(281,191,15,5,colors.white);R(281,196,15,5,'#bd6655');R(281,191,6,5,'#52718a');R(283,192,2,2,colors.white);R(296,192+flutter,5,4,colors.white);R(296,196+flutter,5,4,'#bd6655');
  // Screen activity and learning curves.
  for(let i=0;i<4;i++)R(50,178+i*3,8+((i+Math.floor(t*2))%4)*4,1,i%2?colors.lime:'#82b59c');
  let old=188;
  for(let i=0;i<20;i++){const y=177+Math.min(13,i*.65)+Math.sin(i*.9+t)*1.5;line(132+i*2-2,old,132+i*2,y,'#b1d58c');old=y;}
  for(let i=0;i<4;i++)R(126,284+i*2,9+(i*7+Math.floor(t*2))%19,1,['#a5cbb7','#cbda8c','#d9c295'][i%3]);
  if(Math.floor(t*2)%2)R(144,294,3,1,colors.white);
  seated(91,177,t,colors.blue);seated(177,178,t+1,colors.cream);
  teaching(142,226,t);
  // Students are miniature background characters, distinct from Diego.
  for(let i=0;i<3;i++){R(178+i*25,236,5,4,['#604c38','#455149','#7d6e4b'][i]);R(179+i*25,239,4,3,colors.skinLight);R(177+i*25,242,7,6,[colors.blue,colors.leaf,colors.rust][i]);}
  seated(105,280,t+.3,colors.green);
  // Coffee steam and server LEDs.
  R(110,292-Math.floor(t*3)%3,1,3,'#b9bca4');
  for(let i=0;i<5;i++)if((Math.floor(t*2)+i)%3!==0)R(242,281+i*5,2,1,colors.lime);
  lifting(453,171,t);
  running(558,173,t,colors.skin,.85);
  seated(480,230,t+1,colors.green);seated(570,230,t+.5,colors.blue);
  // A runner on the hill and another Diego catching a quiet reading moment.
  const trailPhase=(t*.045)%1;
  running(302+trailPhase*31,212-trailPhase*25,t,colors.rust,.55);
  head(301,261,true);R(301,270,7,10,colors.cream);R(306,276,3,10,colors.navy);R(306,286,6,2,colors.white);R(302,272,8,5,colors.rust);R(306,272,1,5,colors.cream);
  // Water ripples use discrete frames, so the pool remains pixel-crisp.
  for(let i=0;i<36;i++){const x=341+(i*37+Math.floor(t*4))%245,y=280+(i*11)%33;R(x,y,3+i%5,1,i%2?'#91c9c8':'#68a7b1');}
  swimmer(352+((t*10)%222),283,t);
  swimmer(574-((t*7)%218),298,t+2);
  // Running loop and recovery jog, in different outfits.
  running(zone==='run'?191:(t*15+69)%700-30,329,t,colors.skin,.9);
  running((t*11+389)%700-30,331,t+.5,colors.rust,.8);
  bike(zone==='bike'?210:(t*26+193)%720-40,379,t);
  // Small red Santiago bus on the far side of the bike lane.
  const busX=660-(t*12%790);
  R(busX,358,55,17,'#c16b57');R(busX+2,354,50,4,'#e9e3d2');
  for(let i=0;i<6;i++)R(busX+3+i*8,360,6,6,'#acccbf');
  R(busX,369,55,4,'#e6debe');circle(busX+10,374,3,colors.ink);circle(busX+44,374,3,colors.ink);R(busX+2,368,3,2,'#f0d997');
  // Small ambient details: leaves, glints and a glowing street light at dusk.
  if(dusk||night)for(const x of [13,321,626]){R(x-4,285,10,1,'#efdb9c');R(x-1,286,4,2,'#fff0bf');}
}

const chapters = {
  work: {
    rect:[31,154,230,50], focus:[146,185,2.35], label:'Ciencia de datos', eyebrow:'01 / VIDA LABORAL', title:'De los datos a las decisiones.',
    description:'Trabajo como Data Scientist en Visa Analytics & Consulting. Antes pasé por Falabella y Walmart Chile, desarrollando modelos, pronósticos, recomendaciones y proyectos de machine learning.',
    link:'resume.html', linkLabel:'Mi trayectoria completa ↗'
  },
  teaching: {
    rect:[31,207,230,50], focus:[148,234,2.5], label:'Docencia en la USACH', eyebrow:'02 / DOCENCIA', title:'Lo que aprendo, lo comparto.',
    description:'Desde 2024 soy profesor de Applied Machine Learning en el minor de Data Science de la Universidad de Santiago de Chile. También es donde estudié Ingeniería Civil Industrial y el magíster en Ciencias de la Ingeniería.',
    link:'resume.html', linkLabel:'Más sobre mi formación ↗'
  },
  code: {
    rect:[31,260,230,56], focus:[147,285,2.5], label:'Código y proyectos', eyebrow:'03 / PROYECTOS PERSONALES', title:'Del paper al print().',
    description:'Construyo para entender: implementaciones de papers, aplicaciones con LLMs, visión por computador y experimentos en Python. En este blog comparto el código, el proceso y lo que voy aprendiendo.',
    link:'#blog', linkLabel:'Explorar mis proyectos ↓'
  },
  gym: {
    rect:[430,151,180,54], focus:[522,182,3], label:'Entrenamiento de fuerza', eyebrow:'04 / FUERZA', title:'Una repetición más.',
    description:'El gimnasio también es parte de mi vida. Además de entrenar triatlón, dedico tiempo al trabajo de fuerza. Acá hay otro Diego, lejos del teclado y entre pesas.'
  },
  swim: {
    rect:[332,271,275,53], focus:[468,296,2.2], label:'Natación', eyebrow:'05 / TRIATLÓN · AGUA', title:'Mi otra pantalla es azul.',
    description:'Entreno triatlón, y la natación es una de sus tres disciplinas. Esta es la parte del mundo donde cambio los modelos por la piscina y sigo sumando largos.'
  },
  run: {
    rect:[0,331,640,30], focus:[199,342,2.6], label:'Running', eyebrow:'06 / TRIATLÓN · TIERRA', title:'También pienso en kilómetros.',
    description:'Correr es parte de mi entrenamiento de triatlón. Acá siempre hay un Diego sumando kilómetros, con el cerro y la cordillera de fondo.'
  },
  bike: {
    rect:[0,376,640,28], focus:[223,380,2.7], label:'Ciclismo', eyebrow:'07 / TRIATLÓN · RUEDAS', title:'Un cambio de ritmo.',
    description:'La bicicleta completa el triatlón: nadar, pedalear y correr. Una parte de mi vida que también merecía su propia ciclovía en este pequeño Santiago.'
  }
};
const regions = Object.fromEntries(Object.entries(chapters).map(([key,item])=>[key,[item.rect]]));
function highlight() {
  if(zone==='all')return;
  for(const [x,y,w,h] of regions[zone]) {outline(x-2,y-2,w+4,h+4,colors.lime);outline(x-3,y-3,w+6,h+6,colors.ink);}
}
function draw() {
  c=ctx;ctx.imageSmoothingEnabled=false;drawSky(elapsed);ctx.drawImage(backdrop,0,0);drawLife(elapsed);highlight();
  display.imageSmoothingEnabled=false;
  const width=W/camera.zoom,height=canvas.height/camera.zoom;
  display.clearRect(0,0,W,canvas.height);
  display.drawImage(raster,camera.x-width/2,camera.y-height/2,width,height,0,0,W,canvas.height);
}
function tick(now) {
  frame=0;
  const delta=Math.min((now-previous)/1000,.08);
  if(now-previous>=1000/30) {
    elapsed+=paused?0:delta;previous=now;
    if(cameraMoving) {
      const blend=1-Math.exp(-delta*6);
      for(const key of ['x','y','zoom'])camera[key]+=(cameraTarget[key]-camera[key])*blend;
      if(Math.abs(camera.zoom-cameraTarget.zoom)<.002&&Math.abs(camera.x-cameraTarget.x)<.08&&Math.abs(camera.y-cameraTarget.y)<.08){Object.assign(camera,cameraTarget);cameraMoving=false;}
    }
    draw();
  }
  if((!paused||cameraMoving)&&visible&&!document.hidden)frame=requestAnimationFrame(tick);
}
function wake() {if(!frame&&(!paused||cameraMoving)&&visible&&!document.hidden){previous=performance.now();frame=requestAnimationFrame(tick);}}
function stop() {cancelAnimationFrame(frame);frame=0;}
const worldStage=document.querySelector('.world-stage');
const story=document.getElementById('world-story');
const hotspots=document.getElementById('world-hotspots');
const backButton=document.getElementById('world-back');
for(const [key,chapter] of Object.entries(chapters)) {
  const [x,y,w,h]=chapter.rect;
  const button=document.createElement('button');
  button.type='button';button.className='world-hotspot';button.dataset.chapter=key;button.setAttribute('aria-label',`Explorar: ${chapter.label}`);
  button.style.cssText=`left:${x/W*100}%;top:${y/H*100}%;width:${w/W*100}%;height:${h/H*100}%`;
  button.innerHTML=`<span class="hotspot-marker" aria-hidden="true">+</span><span class="hotspot-name" aria-hidden="true">${chapter.label} ↗</span>`;
  button.addEventListener('click',()=>{setZone(key);backButton.focus({preventScroll:true});});
  hotspots.append(button);
}
function setZone(value) {
  zone=value;
  document.querySelectorAll('[data-world-zone]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.worldZone===zone)));
  const chapter=chapters[zone];
  // A cinematic crop brings the selected room closer and leaves its story in view.
  canvas.height=chapter?288:H;
  hotspots.hidden=Boolean(chapter);backButton.hidden=!chapter;story.hidden=!chapter;
  worldStage.dataset.focused=String(Boolean(chapter));
  if(chapter) {
    document.getElementById('story-eyebrow').textContent=chapter.eyebrow;
    document.getElementById('story-title').textContent=chapter.title;
    document.getElementById('story-text').textContent=chapter.description;
    const link=document.getElementById('story-link');link.hidden=!chapter.link;
    if(chapter.link){link.href=chapter.link;link.textContent=chapter.linkLabel;}
    const [x,y,zoom]=chapter.focus;
    const hw=W/zoom/2,hh=canvas.height/zoom/2;
    cameraTarget={x:Math.max(hw,Math.min(W-hw,x)),y:Math.max(hh,Math.min(H-hh,y)),zoom};
    document.getElementById('world-announcement').textContent=`${chapter.label}. ${chapter.description}`;
  } else {
    cameraTarget={x:W/2,y:H/2,zoom:1};
    document.getElementById('world-announcement').textContent='Vista completa. Elige otra escena para explorar.';
  }
  canvas.setAttribute('aria-label',chapter?`${chapter.label}: ${chapter.description}`:'Vista completa de la vida de Diego en Santiago: trabajo, docencia, código, gimnasio, natación, running y ciclismo.');
  if(motion.matches){Object.assign(camera,cameraTarget);cameraMoving=false;}else cameraMoving=true;
  draw();wake();
}
document.querySelectorAll('[data-world-zone]').forEach(button=>button.addEventListener('click',()=>setZone(button.dataset.worldZone)));
function returnToWorld() {
  const previousZone=zone;
  setZone('all');
  const mobileHome=document.querySelector('[data-world-zone="all"]');
  const target=mobileHome.getClientRects().length?mobileHome:hotspots.querySelector(`[data-chapter="${previousZone}"]`);
  target?.focus({preventScroll:true});
}
backButton.addEventListener('click',returnToWorld);
canvas.addEventListener('click',event=>{
  if(zone!=='all')return;
  const box=canvas.getBoundingClientRect(),x=camera.x-W/camera.zoom/2+(event.clientX-box.left)/box.width*W/camera.zoom,y=camera.y-H/camera.zoom/2+(event.clientY-box.top)/box.height*H/camera.zoom;
  const found=Object.entries(regions).find(([,rectangles])=>rectangles.some(([rx,ry,rw,rh])=>x>=rx&&x<=rx+rw&&y>=ry&&y<=ry+rh));
  setZone(found?.[0]||'all');
});
const pauseButton=document.getElementById('world-pause');
function syncPause() {
  pauseButton.setAttribute('aria-pressed',String(paused));pauseButton.setAttribute('aria-label',paused?'Reanudar animación':'Pausar animación');pauseButton.textContent=paused?'▷':'Ⅱ';
  if(paused)stop();wake();draw();
}
pauseButton.addEventListener('click',()=>{paused=!paused;syncPause();});
motion.addEventListener('change',event=>{paused=event.matches;if(event.matches){Object.assign(camera,cameraTarget);cameraMoving=false;}syncPause();});
const dayButton=document.getElementById('world-day');
const themeButton=document.getElementById('theme-toggle');
function setTimeOfDay(value,persist=true) {
  timeOfDay=value;dusk=value==='dusk';night=value==='night';
  document.documentElement.dataset.theme=night?'night':'day';
  document.querySelector('meta[name="theme-color"]').content=night?'#151e22':'#f4f3ed';
  const label=night?'Activar modo diurno':'Activar modo nocturno';
  themeButton.setAttribute('aria-label',label);themeButton.title=label;
  themeButton.setAttribute('aria-pressed',String(night));
  themeButton.firstElementChild.textContent=night?'☀':'☾';
  themeButton.lastElementChild.textContent=night?'Día':'Noche';
  const next=night?'Ver de día':dusk?'Ver de noche':'Ver al atardecer';
  dayButton.setAttribute('aria-label',next);dayButton.title=next;
  dayButton.textContent=night?'☾':dusk?'◒':'☀';
  if(persist){try{localStorage.setItem('diegulio-time',value);}catch{}}
  bake();draw();
}
dayButton.addEventListener('click',()=>setTimeOfDay(night?'day':dusk?'night':'dusk'));
themeButton.addEventListener('click',()=>setTimeOfDay(night?'day':'night'));

// Move the same canvas into a native dialog: one animation, no duplicate loops.
const dialog=document.getElementById('world-dialog');
const world=document.getElementById('life-world');
const slot=document.getElementById('life-slot');
const expand=document.getElementById('world-expand');
expand.addEventListener('click',()=>{
  if(dialog.open){dialog.close();return;}
  slot.style.minHeight=`${slot.offsetHeight}px`;
  dialog.append(world);dialog.showModal();expand.setAttribute('aria-label','Cerrar vista ampliada');expand.innerHTML='× <span>Cerrar</span>';visible=true;draw();wake();
});
dialog.addEventListener('close',()=>{
  slot.append(world);slot.style.minHeight='';expand.setAttribute('aria-label','Ampliar mundo pixel art');expand.innerHTML='⛶ <span>Ampliar</span>';expand.focus();draw();
});
dialog.addEventListener('click',event=>{if(event.target===dialog){const b=dialog.getBoundingClientRect();if(event.clientX<b.left||event.clientX>b.right||event.clientY<b.top||event.clientY>b.bottom)dialog.close();}});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!dialog.open&&zone!=='all')returnToWorld();});
document.getElementById('story-link').addEventListener('click',()=>{if(dialog.open)dialog.close();});
new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)wake();else stop();},{threshold:.01}).observe(canvas);
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();else wake();});
setTimeOfDay(timeOfDay,false);syncPause();wake();
