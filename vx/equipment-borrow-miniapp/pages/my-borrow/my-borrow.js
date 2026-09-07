const auth = require('../../utils/auth');
const util = require('../../utils/util');

Page({
    data: {
        borrowings: [],
        currentTab: '',
        loading: false,
        page: 1,
        pageSize: 20,
        hasMore: true
    },

    onLoad() {
        this.loadBorrowings();
    },

    onShow() {
        this.setData({
            page: 1,
            hasMore: true,
            borrowings: []
        });
        this.loadBorrowings();
    },

    onPullDownRefresh() {
        this.setData({
            page: 1,
            hasMore: true,
            borrowings: []
        });
        this.loadBorrowings();
        wx.stopPullDownRefresh();
    },

    onReachBottom() {
        if (this.data.hasMore && !this.data.loading) {
            this.loadMore();
        }
    },

    async loadBorrowings() {
        if (this.data.loading) return;

        this.setData({ loading: true });

        try {
            const result = await util.callCloudFunction('borrow', {
                action: 'listMyBorrowings',
                status: this.data.currentTab,
                page: this.data.page,
                pageSize: this.data.pageSize
            });

            if (result.code === 0) {
                const borrowings = result.data.list.map(item => ({
                    ...item,
                    borrowTimeText: util.formatTime(item.borrowTime),
                    expectedReturnTimeText: item.expectedReturnTime
                        ? util.formatDate(item.expectedReturnTime)
                        : '',
                    statusText: this.getStatusText(item.status)
                }));

                const list = this.data.page === 1
                    ? borrowings
                    : [...this.data.borrowings, ...borrowings];

                this.setData({
                    borrowings: list,
                    hasMore: result.data.list.length === this.data.pageSize
                });
            }
        } catch (error) {
            console.error('加载借用列表失败:', error);
            util.showError('加载借用列表失败');
        } finally {
            this.setData({ loading: false });
        }
    },

    getStatusText(status) {
        const statusMap = {
            'borrowing': '借用中',
            'partially_returned': '部分归还',
            'returned': '已归还',
            'cancelled': '已取消'
        };
        return statusMap[status] || status;
    },

    loadMore() {
        this.setData({ page: this.data.page + 1 });
        this.loadBorrowings();
    },

    onTabTap(e) {
        const tab = e.currentTarget.dataset.tab;
        this.setData({
            currentTab: tab,
            page: 1,
            hasMore: true,
            borrowings: []
        });
        this.loadBorrowings();
    },

    onReturnTap(e) {
        const borrow = e.currentTarget.dataset.borrow;
        wx.navigateTo({
            url: `/pages/return/return?borrowRecordId=${borrow._id}&deviceName=${borrow.deviceName}&quantity=${borrow.quantity}&returnedQuantity=${borrow.returnedQuantity}`
        });
    }
});
