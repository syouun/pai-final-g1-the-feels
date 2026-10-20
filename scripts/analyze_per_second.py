import numpy as np, pandas as pd, sys
B="/".join(sys.argv[1:2])
P=sys.argv[1]
chk=pd.read_csv(f"{P}/outputs/check/the_feels_full_g25s_check_frames.csv")
m=np.load(f"{P}/outputs/motions/the_feels_full_g25s.npz")
fps=float(m['fps'][0]); bp=m['body_pos_w']; bq=m['body_quat_w']; T=bp.shape[0]
t=np.arange(T)/fps
# feet = two bodies with lowest mean z
mz=bp[:,:,2].mean(0); feet=np.argsort(mz)[:2]
root=bp[:,0,:]
v=np.gradient(root[:,:2],1/fps,axis=0); sp=np.linalg.norm(v,axis=1)
q=bq[:,0,:] # wxyz
yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
yr=np.abs(np.gradient(yaw,1/fps))
# foot horizontal speed while in contact (h<3cm above that foot's local min)
fs=[];fc=[]
for f in feet:
    h=bp[:,f,2]; base=np.percentile(h,2)
    vf=np.linalg.norm(np.gradient(bp[:,f,:2],1/fps,axis=0),axis=1)
    c=h<base+0.03; fs.append(np.where(c,vf,np.nan)); fc.append(c)
fs=np.nanmean(np.vstack(fs),0) if True else None
both_air=~(fc[0]|fc[1])
df=pd.DataFrame(dict(t=t,root_sp=sp,yaw_rate=yr,foot_slide=fs,air=both_air,root_z=root[:,2]))
df['sec']=df.t.astype(int)
g=df.groupby('sec').agg(root_sp=('root_sp','mean'),root_sp_max=('root_sp','max'),yaw_rate=('yaw_rate','mean'),yaw_max=('yaw_rate','max'),foot_slide=('foot_slide','mean'),air=('air','mean'),root_z=('root_z','mean'))
chk['sec']=chk.time_s.astype(int)
gc=chk.groupby('sec').agg(slip=('slip','sum'),hover=('hover','sum'),selfc=('self_collision','sum'),pen_max=('self_pen_m','max'),Fxy_max=('Fxy_per_mg','max'))
ev={}
for k,f in [('nb4','cmp4_new_policy_g25s.csv'),('nb3','cmp4_prev_policy_g25s.csv')]:
    e=pd.read_csv(f"{P}/outputs/eval/{f}"); e['sec']=e.time_s.astype(int)
    ev[k]=e.groupby('sec').agg(**{f'{k}_jnt':('joint_err_rad','mean'),f'{k}_pos':('body_pos_err_rel_m','mean'),f'{k}_drift':('root_xy_drift_m','max')})
out=g.join(gc).join(ev['nb4']).join(ev['nb3'])
pd.set_option('display.width',300); pd.set_option('display.max_columns',30)
print('feet bodies',feet,'fps',fps)
print(out.loc[0:45].round(3).to_string())
# global ranks
full=out.copy()
print('\nrank by root_sp (top10):',full.root_sp.sort_values(ascending=False).head(10).round(2).to_dict())
print('rank by yaw_rate (top10):',full.yaw_rate.sort_values(ascending=False).head(10).round(2).to_dict())
print('rank by slip (top10):',full.slip.sort_values(ascending=False).head(10).to_dict())
print('rank by nb4_pos (top10):',(full.nb4_pos*100).sort_values(ascending=False).head(10).round(2).to_dict())
print('rank by nb4_jnt (top10):',full.nb4_jnt.sort_values(ascending=False).head(10).round(3).to_dict())
print('corr nb4_pos vs', {c:round(full['nb4_pos'].corr(full[c]),2) for c in ['root_sp','yaw_rate','foot_slide','slip','selfc','pen_max','Fxy_max','air']})
print('corr nb4_jnt vs', {c:round(full['nb4_jnt'].corr(full[c]),2) for c in ['root_sp','yaw_rate','foot_slide','slip','selfc','pen_max','Fxy_max','air']})
out.to_csv(sys.argv[2])
