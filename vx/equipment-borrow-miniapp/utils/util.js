const util = {
    formatTime(date) {
        if (!date) return '';
        const d = new Date(date);
        const year = d.getFullYear();
        const month = (d.getMonth() + 1).toString().padStart(2, '0');
        const day = d.getDate().toString().padStart(2, '0');
        const hour = d.getHours().toString().padStart(2, '0');
        const minute = d.getMinutes().toString().padStart(2, '0');
        return `${year}-${month}-${day} ${hour}:${minute}`;
    },

    formatDate(date) {
        if (!date) return '';
        const d = new Date(date);
        const year = d.getFullYear();
        const month = (d.getMonth() + 1).toString().padStart(2, '0');
        const day = d.getDate().toString().padStart(2, '0');
        return `${year}-${month}-${day}`;
    },

    showLoading(title = '加载中...') {
        wx.showLoading({
            title,
            mask: true
        });
    },

    hideLoading() {
        wx.hideLoading();
    },

    showToast(title, icon = 'none') {
        wx.showToast({
            title,
            icon,
            duration: 2000
        });
    },

    showError(message) {
        wx.showToast({
            title: message,
            icon: 'none',
            duration: 2000
        });
    },

    showSuccess(message) {
        wx.showToast({
            title: message,
            icon: 'success',
            duration: 2000
        });
    },

    async callCloudFunction(name, data) {
        try {
            const result = await wx.cloud.callFunction({
                name,
                data
            });
            return result.result;
        } catch (error) {
            console.error('调用云函数失败:', error);
            return { code: -1, message: '网络错误，请重试' };
        }
    }
};

module.exports = util;
