import matplotlib.pyplot as plt
import time
import numpy   as np
import os

class lp_interior_point():
    def __init__(self,A,b,c):
        #初期化処理
        m = A.shape[1]
        n = A.shape[0]
        np.random.seed(1)
        x0 = np.ones(m)*1000
        y0 = np.zeros(n)
        z0 = np.ones(m)*1000

        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.A = A
        self.b = b
        self.c = c
        return 
    
    def solve(self):
        #主双対内点法による数値解析
        return

    def mu(self,xk,zk): 
        return np.dot(xk,zk)/len(xk)
    
    def w(self,xk,yk,b,c):
        #双対ギャップの算出
        return np.dot(c,xk) - np.dot(b,yk)
    
    def readcsv(self,csvfn):
        #csvファイルの読み出し
        d_ = np.loadtxt(csvfn,delimiter=",")
        return d_

    def savecsv(self,d_,logpath="."):
        #csvファイルの書き出し
        logpath = os.path.join(logpath,"log.csv")
        with open(logpath,"a") as f:
            np.savetxt(f,d_.reshape(1,d_.size), delimiter=",")
        return

    def log(self,xk,yk,zk,muk,wk,dt,logpath="."):
        #主双対内点法による処理を記録
        """
        mu_ | 主変数xと双対変数zの平均値xT*z/N        
        w_  | 双対ギャップcT*x - bT*y
        cx_ | 主問題の目的関数cT*x
        by_ | 双対ex問題の目的関数bT*y
        dt  | 1反復分の計算時間
        x_  | k反復目の主変数xk
        y_  | k反復目の主変数yk
        z_  | k反復目の主変数zk
        """

        x_ = xk.reshape(1,xk.size)[0]
        y_ = yk.reshape(1,yk.size)[0]
        z_ = zk.reshape(1,zk.size)[0]
        mu_= muk.reshape(1,muk.size)[0]
        w_ = wk.reshape(1,wk.size)[0]
        cx_= np.dot(self.c,xk)
        by_= np.dot(self.b,yk)
        d_ = np.hstack([mu_,w_,cx_,by_,dt,x_,y_,z_])
        self.savecsv(d_,logpath)
        return

    def savefig(self,logpath="."):
        #記録した主双対内点法のグラフ化
        csvpath = os.path.join(logpath,"log.csv")
        d_ = self.readcsv(csvpath)
        for i in range(d_.shape[1]):
            plt.figure()
            plt.plot(d_[:,i],"-o")
            plt.grid(True)
            plt.xlabel("iteration")
            pngpath = os.path.join(logpath,"row{0}.png".format(i))
            plt.savefig(pngpath)
            plt.close()

class lp_path(lp_interior_point):
    """
    日付: 2021/03/31
    概要:
    線形計画の主双対内点法の主双対パス追跡法
    
    min  cT*x
    s.t. Ax = b, x>=0
    　　　
    主双対内点法による処理を記録する。
    
    mu_ | 主変数xと双対変数zの平均値xT*z/N
    w_  | 双対ギャップcT*x - bT*y
    dt  | 1反復分の計算時間
    x_  | k反復目の主変数xk
    y_  | k反復目の主変数yk
    z_  | k反復目の主変数zk

    #記録の保存先のパス
    LOGPATH = os.path.join("..","log","lp_proto")
    #線形計画法
    problem = lp_proto(A,b,c)
    #主双対内点法による数値解析
    xk,yk,zk= problem.solve(vervose=True,logpath=LOGPATH)
    #結果の保存
    problem.savefig(path=LOGPATH)
    """

    def lp_path(self,A,b,c,xk,yk,zk):
        #主双対内点法、主双対パス追跡法
        m   = A.shape[1]
        n   = A.shape[0]
        muk = self.mu(xk,zk)
        sigm_k = 0.01

        Xk =np.diag(xk)
        Zk =np.diag(zk)
        e = np.ones(zk.shape)

        r1 = c - np.dot(A.T,yk) - zk
        r2 = b - np.dot(A,xk)
        r3 = (sigm_k*muk)*e -  np.dot(Xk,zk)

        Zk_inv = np.linalg.inv(Zk)
        Xk_inv = np.linalg.inv(Xk)
        
        B     = np.dot( np.dot( np.dot(A,Zk_inv) , Xk), A.T )
        B_inv = np.linalg.inv(B)
        dy  = np.dot(B_inv, np.dot( np.dot(A,Zk_inv), np.dot(Xk,r1) - r3 ) + r2)
        dx  = np.dot( np.dot(Zk_inv,Xk), - r1 + np.dot(Xk_inv,r3) + np.dot(A.T,dy) )
        dz  = np.dot(Xk_inv, r3 - np.dot(Zk,dx))
        
        alpha_p =  np.min([-xi/dxi for xi,dxi in zip(xk,dx)])
        alpha_d =  np.min([-zi/dzi for zi,dzi in zip(zk,dz)])
        tau = 0.95
        alphak  =  min([tau*alpha_p,tau*alpha_d,1])
        if alphak < 0:alphak = 0.1

        xk = xk + alphak*dx
        yk = yk + alphak*dy
        zk = zk + alphak*dz

        return xk,yk,zk
    
    
    def solve(self,vervose=False,logpath=".",eps=1e-3):
        #主双対内点法による数値解析
        t0 = time.time()
        A  = self.A
        b  = self.b
        c  = self.c

        xk = self.x0
        yk = self.y0
        zk = self.z0

        muk= self.mu(xk,zk)
        wk = self.w(xk,yk,b,c)
        
        t1 = time.time()
        dt = t1 -t0

        if vervose == True:
            csvpath = os.path.join(logpath,"log.csv")
            if os.path.isfile(csvpath):
                os.remove(csvpath)
            self.log(xk,yk,zk,muk,wk,dt,logpath)

        for i in range(100):
            t0 = time.time()
            xk,yk,zk = self.lp_path(A,b,c,xk,yk,zk)
            muk= self.mu(xk,zk)
            wk = self.w(xk,yk,b,c)
            t1 = time.time()
            dt = t1 -t0
            if vervose == True:
                self.log(xk,yk,zk,muk,wk,dt,logpath)
            if np.abs(muk) < eps:
                break
        return xk,yk,zk


class lp_affine(lp_interior_point):
    """
    日付: 2021/03/31
    概要:
    線形計画の主双対内点法の主双対アフィンスケーリング法
    
    min  cT*x
    s.t. Ax = b, x>=0
    　　　
    主双対内点法による処理を記録する。
    
    mu_ | 主変数xと双対変数zの平均値xT*z/N
    w_  | 双対ギャップcT*x - bT*y
    dt  | 1反復分の計算時間
    x_  | k反復目の主変数xk
    y_  | k反復目の主変数yk
    z_  | k反復目の主変数zk

    #記録の保存先のパス
    LOGPATH = os.path.join("..","log","lp_proto")
    #線形計画法
    problem = lp_proto(A,b,c)
    #主双対内点法による数値解析
    xk,yk,zk= problem.solve(vervose=True,logpath=LOGPATH)
    #結果の保存
    problem.savefig(path=LOGPATH)
    """
    def lp_affine(self,A,b,c,xk,yk,zk):
        #主双対内点法、主双対アフィンスケーリング法
        m   = A.shape[1]
        n   = A.shape[0]
        muk = self.mu(xk,zk)
        
        Xk =np.diag(xk)
        Zk =np.diag(zk)
        e = np.ones(zk.shape)

        r1 = c - np.dot(A.T,yk) - zk
        r2 = b - np.dot(A,xk)
        r3 = -  np.dot(Xk,zk)

        Zk_inv = np.linalg.inv(Zk)
        Xk_inv = np.linalg.inv(Xk)
        
        B = np.dot( np.dot( np.dot(A,Zk_inv) , Xk), A.T )
        B_inv = np.linalg.inv(B)
        dy  = np.dot(B_inv, np.dot( np.dot(A,Zk_inv), np.dot(Xk,r1) - r3 ) + r2)
        dx  = np.dot( np.dot(Zk_inv,Xk), - r1 + np.dot(Xk_inv,r3) + np.dot(A.T,dy) )
        dz  = np.dot(Xk_inv, r3 - np.dot(Zk,dx))
        
        alpha_p =  np.min([-xi/dxi for xi,dxi in zip(xk,dx)])
        alpha_d =  np.min([-zi/dzi for zi,dzi in zip(zk,dz)])
        tau = 0.95
        alphak  =  min([tau*alpha_p,tau*alpha_d,1])
        if alphak < 0:alphak = 0.1

        xk = xk + alphak*dx
        yk = yk + alphak*dy
        zk = zk + alphak*dz
        
        return xk,yk,zk
    
    def solve(self,vervose=False,logpath=".",eps=1e-3):
        #主双対内点法による数値解析
        t0 = time.time()
        A  = self.A
        b  = self.b
        c  = self.c

        xk = self.x0
        yk = self.y0
        zk = self.z0

        muk= self.mu(xk,zk)
        wk = self.w(xk,yk,b,c)
        
        t1 = time.time()
        dt = t1 -t0

        if vervose == True:
            csvpath = os.path.join(logpath,"log.csv")
            if os.path.isfile(csvpath):
                os.remove(csvpath)
            self.log(xk,yk,zk,muk,wk,dt,logpath)

        for i in range(100):
            t0 = time.time()
            xk,yk,zk = self.lp_affine(A,b,c,xk,yk,zk)
            muk= self.mu(xk,zk)
            wk = self.w(xk,yk,b,c)
            t1 = time.time()
            dt = t1 -t0
            if vervose == True:
                self.log(xk,yk,zk,muk,wk,dt,logpath)
            if np.abs(muk) < eps:
                break
        return xk,yk,zk

class lp_potential(lp_interior_point):
    """
    日付: 2021/03/31
    概要:
    線形計画の主双対内点法の主双対ポテンシャル減少法
    
    min  cT*x
    s.t. Ax = b, x>=0
    　　　
    主双対内点法による処理を記録する。
    
    mu_ | 主変数xと双対変数zの平均値xT*z/N
    w_  | 双対ギャップcT*x - bT*y
    dt  | 1反復分の計算時間
    x_  | k反復目の主変数xk
    y_  | k反復目の主変数yk
    z_  | k反復目の主変数zk

    #記録の保存先のパス
    LOGPATH = os.path.join("..","log","lp_proto")
    #線形計画法
    problem = lp_proto(A,b,c)
    #主双対内点法による数値解析
    xk,yk,zk= problem.solve(vervose=True,logpath=LOGPATH)
    #結果の保存
    problem.savefig(path=LOGPATH)
    """
    def lp_potential(self,A,b,c,xk,yk,zk):
        #主双対内点法、主双対ポテンシャル減少法
        m   = A.shape[1]
        n   = A.shape[0]
        muk = self.mu(xk,zk)
        sigm_k = 0.01

        Xk =np.diag(xk)
        Zk =np.diag(zk)
        e = np.ones(zk.shape)

        r1 = c - np.dot(A.T,yk) - zk
        r2 = b - np.dot(A,xk)
        r3 = (sigm_k*muk)*e -  np.dot(Xk,zk)

        Zk_inv = np.linalg.inv(Zk)
        Xk_inv = np.linalg.inv(Xk)
        
        B = np.dot( np.dot( np.dot(A,Zk_inv) , Xk), A.T )
        B_inv = np.linalg.inv(B)
        dy  = np.dot(B_inv, np.dot( np.dot(A,Zk_inv), np.dot(Xk,r1) - r3 ) + r2)
        dx  =  np.dot( np.dot(Zk_inv,Xk), - r1 + np.dot(Xk_inv,r3) + np.dot(A.T,dy) )
        dz  = np.dot(Xk_inv, r3 - np.dot(Zk,dx))
        
        def f_nu(xk,zk,alpha):
            nu = 1
            f1 =  nu * np.log(np.dot(xk,zk)) 
            f2 = len(xk) * np.log(np.dot(xk,zk)/len(xk))
            f3 = 0
            for xi,zi in zip(xk,zk):
                f3 = f3 + np.log(xi*zi)
            return f1 + f2 - f3
        
        F = []
        Alpha = np.linspace(0.01,1,5)
        for alphak in Alpha:
            xk_ = xk + alphak*dx
            zk_ = zk + alphak*dz
            F.append(f_nu(xk_,zk_,alphak))
        
        indF = F.index(min(F))
        alphak = Alpha[indF]

        xk = xk + alphak*dx
        yk = yk + alphak*dy
        zk = zk + alphak*dz

        return xk,yk,zk

    def solve(self,vervose=False,logpath=".",eps=1e-3):
        #主双対内点法による数値解析
        t0 = time.time()
        A  = self.A
        b  = self.b
        c  = self.c

        xk = self.x0
        yk = self.y0
        zk = self.z0

        muk= self.mu(xk,zk)
        wk = self.w(xk,yk,b,c)
        
        t1 = time.time()
        dt = t1 -t0

        if vervose == True:
            csvpath = os.path.join(logpath,"log.csv")
            if os.path.isfile(csvpath):
                os.remove(csvpath)
            self.log(xk,yk,zk,muk,wk,dt,logpath)

        for i in range(100):
            t0 = time.time()
            xk,yk,zk = self.lp_potential(A,b,c,xk,yk,zk)
            muk= self.mu(xk,zk)
            wk = self.w(xk,yk,b,c)
            t1 = time.time()
            dt = t1 -t0
            if vervose == True:
                self.log(xk,yk,zk,muk,wk,dt,logpath)
            if np.abs(muk) < eps:
                break
        return xk,yk,zk

if __name__ == "__main__":
    
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
    