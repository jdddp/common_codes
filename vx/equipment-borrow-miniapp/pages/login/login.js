const auth = require('../../utils/auth');
const util = require('../../utils/util');

Page({
    data: {
        phone: '',
        password: '',
        showPassword: false,
        loading: false
    },

    onLoad() {
        if (auth.isLoggedIn()) {
            this.redirectToIndex();
        }
    },

    onPhoneInput(e) {
        this.setData({ phone: e.detail.value });
    },

    onPasswordInput(e) {
        this.setData({ password: e.detail.value });
    },

    togglePassword() {
        this.setData({ showPassword: !this.data.showPassword });
    },

    async handleLogin() {
        const { phone, password } = this.data;

        if (!phone) {
            util.showError('请输入手机号');
            return;
        }

        if (phone.length !== 11) {
            util.showError('手机号格式不正确');
            return;
        }

        if (!password) {
            util.showError('请输入密码');
            return;
        }

        this.setData({ loading: true });

        try {
            const result = await util.callCloudFunction('auth', {
                action: 'login',
                phone,
                password
            });

            if (result.code === 0) {
                auth.setAuth(result.data.token, result.data.userInfo);
                util.showSuccess('登录成功');
                setTimeout(() => {
                    this.redirectToIndex();
                }, 1500);
            } else {
                util.showError(result.message || '登录失败');
            }
        } catch (error) {
            util.showError('网络错误，请重试');
        } finally {
            this.setData({ loading: false });
        }
    },

    redirectToIndex() {
        const userInfo = auth.getUserInfo();
        if (userInfo && userInfo.role === 'admin') {
            wx.switchTab({
                url: '/pages/index/index'
            });
        } else {
            wx.switchTab({
                url: '/pages/index/index'
            });
        }
    }
});
