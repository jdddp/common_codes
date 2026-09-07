const auth = {
    token: null,
    userInfo: null,

    init() {
        try {
            const token = wx.getStorageSync('token');
            const userInfo = wx.getStorageSync('userInfo');
            if (token) {
                this.token = token;
            }
            if (userInfo) {
                this.userInfo = userInfo;
            }
        } catch (e) {
            console.error('初始化认证信息失败:', e);
        }
    },

    setAuth(token, userInfo) {
        this.token = token;
        this.userInfo = userInfo;
        wx.setStorageSync('token', token);
        wx.setStorageSync('userInfo', userInfo);
    },

    clearAuth() {
        this.token = null;
        this.userInfo = null;
        wx.removeStorageSync('token');
        wx.removeStorageSync('userInfo');
    },

    getToken() {
        return this.token;
    },

    getUserInfo() {
        return this.userInfo;
    },

    isLoggedIn() {
        return !!this.token;
    },

    isAdmin() {
        return this.userInfo && this.userInfo.role === 'admin';
    },

    checkLogin() {
        if (!this.isLoggedIn()) {
            wx.redirectTo({
                url: '/pages/login/login'
            });
            return false;
        }
        return true;
    },

    checkAdmin() {
        if (!this.checkLogin()) {
            return false;
        }
        if (!this.isAdmin()) {
            wx.showToast({
                title: '无管理员权限',
                icon: 'none'
            });
            return false;
        }
        return true;
    }
};

auth.init();

module.exports = auth;
