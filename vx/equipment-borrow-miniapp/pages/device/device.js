const auth = require('../../utils/auth');
const util = require('../../utils/util');

Page({
    data: {
        categories: [],
        devices: [],
        currentCategory: '',
        keyword: '',
        loading: false,
        page: 1,
        pageSize: 20,
        hasMore: true
    },

    onLoad() {
        this.loadCategories();
        this.loadDevices();
    },

    onPullDownRefresh() {
        this.setData({
            page: 1,
            hasMore: true,
            devices: []
        });
        this.loadDevices();
        wx.stopPullDownRefresh();
    },

    onReachBottom() {
        if (this.data.hasMore && !this.data.loading) {
            this.loadMore();
        }
    },

    async loadCategories() {
        try {
            const result = await util.callCloudFunction('device', {
                action: 'listCategories'
            });

            if (result.code === 0) {
                this.setData({ categories: result.data });
            }
        } catch (error) {
            console.error('加载分类失败:', error);
        }
    },

    async loadDevices() {
        if (this.data.loading) return;

        this.setData({ loading: true });

        try {
            const result = await util.callCloudFunction('device', {
                action: 'listDevices',
                categoryId: this.data.currentCategory,
                page: this.data.page,
                pageSize: this.data.pageSize
            });

            if (result.code === 0) {
                const devices = this.data.page === 1
                    ? result.data.list
                    : [...this.data.devices, ...result.data.list];

                this.setData({
                    devices,
                    hasMore: result.data.list.length === this.data.pageSize
                });
            }
        } catch (error) {
            console.error('加载设备失败:', error);
            util.showError('加载设备失败');
        } finally {
            this.setData({ loading: false });
        }
    },

    loadMore() {
        this.setData({ page: this.data.page + 1 });
        this.loadDevices();
    },

    onSearchInput(e) {
        this.setData({ keyword: e.detail.value });
    },

    onSearch() {
        this.setData({ page: 1, hasMore: true, devices: [] });
        this.loadDevices();
    },

    onCategoryTap(e) {
        const categoryId = e.currentTarget.dataset.id;
        this.setData({
            currentCategory: categoryId,
            page: 1,
            hasMore: true,
            devices: []
        });
        this.loadDevices();
    },

    onDeviceTap(e) {
        const device = e.currentTarget.dataset.device;
        if (device.status === 'active' && device.availableQuantity > 0) {
            this.onBorrowTap(e);
        }
    },

    onBorrowTap(e) {
        const device = e.currentTarget.dataset.device;
        wx.navigateTo({
            url: `/pages/borrow/borrow?deviceId=${device._id}&deviceName=${device.name}&availableQuantity=${device.availableQuantity}`
        });
    }
});
