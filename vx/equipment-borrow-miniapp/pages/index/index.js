const auth = require('../../utils/auth');

Page({
    data: {
        userInfo: null,
        isAdmin: false
    },

    onLoad() {
        this.loadUserInfo();
    },

    onShow() {
        this.loadUserInfo();
    },

    loadUserInfo() {
        const userInfo = auth.getUserInfo();
        this.setData({
            userInfo,
            isAdmin: auth.isAdmin()
        });
    },

    goToDevice() {
        wx.switchTab({
            url: '/pages/device/device'
        });
    },

    goToMyBorrow() {
        wx.switchTab({
            url: '/pages/my-borrow/my-borrow'
        });
    },

    goToRecords() {
        wx.navigateTo({
            url: '/pages/records/records'
        });
    },

    goToPassword() {
        wx.switchTab({
            url: '/pages/password/password'
        });
    },

    goToAdminDevice() {
        wx.navigateTo({
            url: '/pages/admin/device/device'
        });
    },

    goToAdminUser() {
        wx.navigateTo({
            url: '/pages/admin/user/user'
        });
    },

    goToAdminRecord() {
        wx.navigateTo({
            url: '/pages/admin/record/record'
        });
    }
});
