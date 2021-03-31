# linear-programming
線形計画問題を主双対内点法により解く。主双対内点法のうち主双対アフィンスケーリング法、主双対パス追跡法、主双対ポテンシャル減少法が実装されている。

使用例

```
    from lplib import lp_path, lp_affine,lp_potential
    import os
    
    A = np.array([[5,2],[1,2]])
    b = np.array([30,14])
    c = np.array([-5,-4])
    
    #A = np.array([[2,10,4,1,0,0],[6,5,8,0,1,0],[7,10,8,0,0,1]])
    #b = np.array([425,400,600])
    #c = np.array([-2.5,-5,-3.4,0,0,0])
    
    #主双対パス追跡法
    #記録の保存先のパス
    LOGPATH = os.path.join("..","log","lp_path")
    if not os.path.isdir(LOGPATH):
        os.mkdir(LOGPATH)
    #線形計画法の主双対パス追跡法
    problem = lp_path(A,b,c)
    #数値解析
    xk,yk,zk= problem.solve(vervose=True,logpath=LOGPATH)
    #結果の保存
    problem.savefig(logpath=LOGPATH)

    
    #主双対アフィンスケーリング法
    #記録の保存先のパス
    LOGPATH = os.path.join("..","log","lp_affine")
    if not os.path.isdir(LOGPATH):
        os.mkdir(LOGPATH)
    #線形計画法の主双対アフィンスケーリング法
    problem = lp_affine(A,b,c)
    #数値解析
    xk,yk,zk= problem.solve(vervose=True,logpath=LOGPATH)
    #結果の保存
    problem.savefig(logpath=LOGPATH)

    #主双対ポテンシャル減少法法
    #記録の保存先のパス
    LOGPATH = os.path.join("..","log","lp_potential")
    if not os.path.isdir(LOGPATH):
        os.mkdir(LOGPATH)
    #線形計画法の主双対ポテンシャル減少法法
    problem = lp_potential(A,b,c)
    #数値解析
    xk,yk,zk= problem.solve(vervose=True,logpath=LOGPATH)
    #結果の保存
    problem.savefig(logpath=LOGPATH)
    
   ```
