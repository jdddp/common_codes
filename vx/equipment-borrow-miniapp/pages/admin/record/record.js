const auth = require('../../../utils/auth');
const util = require('../../../utils/util');

Page({
    data: {
        records: [],
        statusIndex: 0,
        statusOptions: [
            { label: '借用中', value: 'borrowing' },
            { label: '已归还', value: 'returned' }
        ],
        startDate: '',
        endDate: '',
        loading: false,
        page: 1,
        pageSize: 20,
        hasMore: true
    },

    onLoad() {},

    onShow() {
        if (!auth.checkAdmin()) return;
        this.resetAndLoad();
    },

    onPullDownRefresh() {
        this.resetAndLoad();
        wx.stopPullDownRefresh();
    },

    onReachBottom() {
        if (this.data.hasMore && !this.data.loading) {
            this.loadMore();
        }
    },

    resetAndLoad() {
        this.setData({
            page: 1,
            hasMore: true,
            records: [],
            loading: false
        });
        this.loadRecords();
    },

    async loadRecords() {
        if (this.data.loading) return;

        this.setData({ loading: true });

        try {
            const result = await util.callCloudFunction('record', {
                action: 'listAllRecords',
                status: this.data.statusOptions[this.data.statusIndex].value,
                startDate: this.data.startDate,
                endDate: this.data.endDate,
                page: this.data.page,
                pageSize: this.data.pageSize
            });

            if (result.code === 0) {
                const records = result.data.list.map(item => ({
                    ...item,
                    borrowTimeText: util.formatTime(item.borrowTime),
                    statusText: this.getStatusText(item.status)
                }));

                const list = this.data.page === 1
                    ? records
                    : [...this.data.records, ...records];

                this.setData({
                    records: list,
                    hasMore: result.data.list.length === this.data.pageSize
                });
            }
        } catch (error) {
            console.error('加载记录失败:', error);
            util.showError('加载记录失败');
        } finally {
            this.setData({ loading: false });
        }
    },

    getStatusText(status) {
        const statusMap = {
            'borrowing': '借用中',
            'partially_returned': '借用中',
            'returned': '已归还'
        };
        return statusMap[status] || status;
    },

    loadMore() {
        this.setData({ page: this.data.page + 1 });
        this.loadRecords();
    },

    onStatusChange(e) {
        this.setData({ statusIndex: e.detail.value });
        this.resetAndLoad();
    },

    onStartDateChange(e) {
        this.setData({ startDate: e.detail.value });
        this.resetAndLoad();
    },

    onEndDateChange(e) {
        this.setData({ endDate: e.detail.value });
        this.resetAndLoad();
    }
});
