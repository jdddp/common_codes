const util = require('../../utils/util');

Page({
    data: {
        borrowRecordId: '',
        deviceName: '',
        quantity: 0,
        returnedQuantity: 0,
        outstandingQuantity: 0,
        returnQuantity: '',
        conditionIndex: 0,
        conditionOptions: [
            { label: '正常', value: 'normal' },
            { label: '损坏', value: 'damaged' },
            { label: '丢失', value: 'lost' }
        ],
        remark: '',
        loading: false
    },

    onLoad(options) {
        const quantity = parseInt(options.quantity) || 0;
        const returnedQuantity = parseInt(options.returnedQuantity) || 0;
        const outstandingQuantity = quantity - returnedQuantity;

        this.setData({
            borrowRecordId: options.borrowRecordId || '',
            deviceName: options.deviceName || '',
            quantity,
            returnedQuantity,
            outstandingQuantity
        });
    },

    onQuantityInput(e) {
        this.setData({ returnQuantity: e.detail.value });
    },

    onConditionChange(e) {
        this.setData({ conditionIndex: e.detail.value });
    },

    onRemarkInput(e) {
        this.setData({ remark: e.detail.value });
    },

    async onSubmit() {
        const { borrowRecordId, returnQuantity, conditionOptions, conditionIndex, remark, outstandingQuantity } = this.data;

        if (!returnQuantity || returnQuantity <= 0) {
            util.showError('请输入归还数量');
            return;
        }

        if (parseInt(returnQuantity) > outstandingQuantity) {
            util.showError(`归还数量不能超过待归还数量（${outstandingQuantity}）`);
            return;
        }

        this.setData({ loading: true });

        try {
            const result = await util.callCloudFunction('return', {
                action: 'returnDevice',
                borrowRecordId,
                quantity: parseInt(returnQuantity),
                condition: conditionOptions[conditionIndex].value,
                remark
            });

            if (result.code === 0) {
                util.showSuccess('归还成功');
                setTimeout(() => {
                    wx.navigateBack();
                }, 1500);
            } else {
                util.showError(result.message || '归还失败');
            }
        } catch (error) {
            util.showError('网络错误，请重试');
        } finally {
            this.setData({ loading: false });
        }
    }
});
