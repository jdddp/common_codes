const util = require('../../utils/util');

Page({
    data: {
        deviceId: '',
        deviceName: '',
        availableQuantity: 0,
        quantity: '',
        expectedReturnTime: '',
        purpose: '',
        remark: '',
        today: '',
        loading: false
    },

    onLoad(options) {
        const today = util.formatDate(new Date());
        this.setData({
            deviceId: options.deviceId || '',
            deviceName: decodeURIComponent(options.deviceName || ''),
            availableQuantity: parseInt(options.availableQuantity) || 0,
            today
        });
    },

    onQuantityInput(e) {
        this.setData({ quantity: e.detail.value });
    },

    onDateChange(e) {
        this.setData({ expectedReturnTime: e.detail.value });
    },

    onPurposeInput(e) {
        this.setData({ purpose: e.detail.value });
    },

    onRemarkInput(e) {
        this.setData({ remark: e.detail.value });
    },

    async onSubmit() {
        const { deviceId, quantity, expectedReturnTime, purpose, remark, availableQuantity } = this.data;

        if (!quantity || quantity <= 0) {
            util.showError('请输入借用数量');
            return;
        }

        if (parseInt(quantity) > availableQuantity) {
            util.showError(`借用数量不能超过可用数量（${availableQuantity}）`);
            return;
        }

        this.setData({ loading: true });

        try {
            const result = await util.callCloudFunction('borrow', {
                action: 'borrowDevice',
                deviceTypeId: deviceId,
                quantity: parseInt(quantity),
                expectedReturnTime: expectedReturnTime || null,
                purpose,
                remark
            });

            if (result.code === 0) {
                util.showSuccess('借用成功');
                setTimeout(() => {
                    wx.navigateBack();
                }, 1500);
            } else {
                util.showError(result.message || '借用失败');
            }
        } catch (error) {
            util.showError('网络错误，请重试');
        } finally {
            this.setData({ loading: false });
        }
    }
});
