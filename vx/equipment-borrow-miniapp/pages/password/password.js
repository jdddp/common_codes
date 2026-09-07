const auth = require('../../utils/auth');
const util = require('../../utils/util');

Page({
    data: {
        oldPassword: '',
        newPassword: '',
        confirmPassword: '',
        showNewPassword: false,
        showConfirmPassword: false,
        loading: false
    },

    onOldPasswordInput(e) {
        this.setData({ oldPassword: e.detail.value });
    },

    onNewPasswordInput(e) {
        this.setData({ newPassword: e.detail.value });
    },

    onConfirmPasswordInput(e) {
        this.setData({ confirmPassword: e.detail.value });
    },

    toggleNewPassword() {
        this.setData({ showNewPassword: !this.data.showNewPassword });
    },

    toggleConfirmPassword() {
        this.setData({ showConfirmPassword: !this.data.showConfirmPassword });
    },

    async onSubmit() {
        const { oldPassword, newPassword, confirmPassword } = this.data;

        if (!oldPassword) {
            util.showError('请输入旧密码');
            return;
        }

        if (!newPassword) {
            util.showError('请输入新密码');
            return;
        }

        if (newPassword.length < 8) {
            util.showError('新密码至少8位');
            return;
        }

        if (newPassword !== confirmPassword) {
            util.showError('两次输入的密码不一致');
            return;
        }

        this.setData({ loading: true });

        try {
            const result = await util.callCloudFunction('auth', {
                action: 'changePassword',
                oldPassword,
                newPassword
            });

            if (result.code === 0) {
                auth.clearAuth();
                util.showSuccess('密码修改成功，请重新登录');
                setTimeout(() => {
                    wx.redirectTo({
                        url: '/pages/login/login'
                    });
                }, 1500);
            } else {
                util.showError(result.message || '修改失败');
            }
        } catch (error) {
            util.showError('网络错误，请重试');
        } finally {
            this.setData({ loading: false });
        }
    },

    onLogout() {
        wx.showModal({
            title: '提示',
            content: '确定要退出登录吗？',
            success: (res) => {
                if (res.confirm) {
                    auth.clearAuth();
                    wx.redirectTo({
                        url: '/pages/login/login'
                    });
                }
            }
        });
    }
});
