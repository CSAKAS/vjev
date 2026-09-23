"""Deterministic RGB-only games. Oracle state is never accepted by the policy."""
import random
from collections import deque
from io import BytesIO
from PIL import Image, ImageDraw

ACTIONS={'UP':'Choose or move toward the top of the image.',
         'DOWN':'Choose or move toward the bottom of the image.',
         'LEFT':'Choose or move toward the left of the image.',
         'RIGHT':'Choose or move toward the right of the image.'}
DELTAS={'UP':(0,-1),'DOWN':(0,1),'LEFT':(-1,0),'RIGHT':(1,0)}
SPECS={
 'navigation':{'name':'无障碍导航','task':'Move the red square into the green goal using as few moves as possible. Each move changes one grid cell. The square cannot leave the grid.','steps':20},
 'detour':{'name':'绕墙导航','task':'Move the red square into the green goal using as few moves as possible. Each move changes one grid cell. Dark gray cells are walls: you cannot enter or cross them. You cannot leave the grid. Plan around walls.','steps':20},
 'color_match':{'name':'颜色匹配','task':'Choose the outer square whose color matches the central circle. Answer the direction of that square relative to the central circle.','steps':1},
 'count_max':{'name':'数量比较','task':'Four panels contain circles. Choose the panel with the greatest number of circles. Answer the direction of that panel relative to the image center.','steps':1},
}
SYSTEM='Solve the visual task using the current RGB image. Directions always refer to the image. Follow the requested answer format.'
CENTERS={'UP':(256,90),'DOWN':(256,422),'LEFT':(90,256),'RIGHT':(422,256)}
PALETTE=[(231,62,69),(47,116,219),(48,164,96),(239,172,35)]


def png(image):
    stream=BytesIO();image.save(stream,format='PNG');return stream.getvalue()


class Game:
    def __init__(self,kind,seed):
        self.kind=kind;self.seed=seed;self.done=False;self.success=False
        rng=random.Random(seed)
        self.walls=set()
        if kind in ('navigation','detour'):
            for _ in range(10000):
                self.position=(rng.randrange(6),rng.randrange(6));self.goal=(rng.randrange(6),rng.randrange(6))
                if self.position==self.goal:continue
                available=[(x,y) for y in range(6) for x in range(6) if (x,y) not in (self.position,self.goal)]
                self.walls=set(rng.sample(available,7)) if kind=='detour' else set()
                distances=self.distances();distance=distances.get(self.position)
                manhattan=sum(abs(x-y) for x,y in zip(self.position,self.goal))
                if distance is not None and 3<=distance<=10 and (kind=='navigation' or distance>manhattan):break
            else:raise ValueError('Could not generate solvable scene')
            self.shortest=distance
        elif kind=='color_match':
            self.colors=list(range(4));rng.shuffle(self.colors);self.target_color=rng.randrange(4)
            self.correct=list(ACTIONS)[self.colors.index(self.target_color)];self.shortest=1
        elif kind=='count_max':
            self.counts=rng.sample(range(1,10),4)
            self.slots=[rng.sample(range(9),n) for n in self.counts]
            self.correct=list(ACTIONS)[self.counts.index(max(self.counts))];self.shortest=1
        else:raise ValueError(kind)

    def target(self,action):
        dx,dy=DELTAS[action];x,y=self.position;target=(x+dx,y+dy)
        return target if 0<=target[0]<6 and 0<=target[1]<6 and target not in self.walls else self.position

    def distances(self):
        result={self.goal:0};queue=deque([self.goal])
        while queue:
            x,y=queue.popleft()
            for dx,dy in DELTAS.values():
                p=(x+dx,y+dy)
                if 0<=p[0]<6 and 0<=p[1]<6 and p not in self.walls and p not in result:
                    result[p]=result[(x,y)]+1;queue.append(p)
        return result

    def optimal(self):
        if self.kind not in ('navigation','detour'):return [self.correct]
        distances=self.distances();before=distances[self.position]
        return [a for a in ACTIONS if distances.get(self.target(a),999)==before-1]

    def state(self):
        # Output/evaluation-only; this object never enters the policy API.
        if self.kind in ('navigation','detour'):
            return {'position':list(self.position),'goal':list(self.goal),'walls':sorted(map(list,self.walls)),
                    'distance':self.distances()[self.position]}
        if self.kind=='color_match':return {'colors':self.colors,'target_color':self.target_color,'correct':self.correct}
        return {'counts':self.counts,'slots':self.slots,'correct':self.correct}

    def step(self,action):
        if self.done:raise ValueError('Episode has ended')
        if action not in ACTIONS:raise ValueError('Invalid action; no fallback')
        before=self.state();correct=action in self.optimal()
        if self.kind in ('navigation','detour'):
            new=self.target(action);blocked=new==self.position;self.position=new
            self.success=self.position==self.goal;self.done=self.success
        else:
            blocked=False;self.success=correct;self.done=True
        return {'before':before,'after':self.state(),'action':action,'optimal':correct,'blocked':blocked,'success':self.success}

    def render(self):
        image=Image.new('RGB',(512,512),(246,247,250));draw=ImageDraw.Draw(image)
        if self.kind in ('navigation','detour'):
            margin,cell=28,76
            for i in range(7):
                t=margin+i*cell
                draw.line((t,margin,t,margin+6*cell),fill=(190,195,203),width=2)
                draw.line((margin,t,margin+6*cell,t),fill=(190,195,203),width=2)
            def bounds(p,pad):
                x,y=p;return (margin+x*cell+pad,margin+y*cell+pad,margin+(x+1)*cell-pad,margin+(y+1)*cell-pad)
            for wall in sorted(self.walls):draw.rectangle(bounds(wall,2),fill=(71,77,87))
            draw.rectangle(bounds(self.goal,6),fill=(69,177,112),outline=(24,107,63),width=3)
            draw.rectangle(bounds(self.position,16),fill=(231,62,69),outline=(144,31,39),width=3)
        elif self.kind=='color_match':
            draw.ellipse((220,220,292,292),fill=PALETTE[self.target_color])
            for (x,y),color in zip(CENTERS.values(),self.colors):draw.rectangle((x-43,y-43,x+43,y+43),fill=PALETTE[color])
        else:
            for (x,y),slots in zip(CENTERS.values(),self.slots):
                draw.rectangle((x-71,y-71,x+71,y+71),outline=(140,145,153),width=2)
                for slot in slots:
                    cx=x+(slot%3-1)*39;cy=y+(slot//3-1)*39
                    draw.ellipse((cx-10,cy-10,cx+10,cy+10),fill=(42,91,180))
        return png(image)


def cases(n=50):
    result=[]
    for kind in SPECS:
        seen=set();seed=2026092200
        while len(seen)<n:
            game=Game(kind,seed);rgb=game.render()
            if rgb not in seen:
                seen.add(rgb);result.append({'game':kind,'seed':seed,'index':len(seen)-1})
            seed+=1
    return result
