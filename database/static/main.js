const mixin = {
  data () {
    return {
      socket: io(),
      collection: '',
      collectionList: [],

      showBox: 1,
      suffix: 'ed-v',
    }
  },
  delimiters: ["[[", "]]"],
  filters: {
    round: function(value, decimals) {
      if (!value) { value = 0; }
      if (!decimals) { decimals = 0; }
      return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
    },
    truncate: function (text, length, clamp) {
      text = text || '';
      length = length || 30;
      clamp = clamp || '...';

      if (text.length <= length) return text;
      tcText =  text.slice(text.length - length - 1 + clamp.length, text.length - 1);
      return clamp + tcText;
    },
  },
  methods: {
    updateCollections: function(event) {
      return axios.get('api/collection-list').then(response => {
        this.collectionList = response.data;
        this.collection = this.collection || this.collectionList[0];
      });
    }
  }
}

const Overview = Vue.extend({
  template: '#overview-template',
  data () {
    return {
      pageIndex: 0,
      pageCount: 0,
      pageData: [],

      rowCount: 4,
      columnCount: 4,

      currentEpisodeId: '',
      currentActionId: 0,
      detailEpisode: null,
      detailAction: null,

      stats: undefined,
      filter: {
        query: '',
      },
      showPageIndexInput: false,
      showSecondImage: false,
      showSettings: false,
      showAdvanced: false,
    }
  },
  created () {
    if (this.$route.query.page) {
      this.pageIndex = parseInt(this.$route.query.page);
    }

    this.collection = localStorage.getItem('collection') || this.collection;
    this.rowCount = Math.max(localStorage.getItem('row_count') || this.rowCount, 1);
    this.columnCount = Math.max(localStorage.getItem('column_count') || this.columnCount, 1);
    this.showBox = localStorage.getItem('show_box') || this.showBox;
    this.suffix = localStorage.getItem('suffix') || this.suffix;

    this.updateCollections().then(response => {
      this.updateActions();
    });

    this.socket.on('new-episode', data => {
      if (data.collection !== this.collection) return;

      this.updateStats();

      if (this.pageIndex === 0) {
        if (this.filter.query && this.filter.query !== '') {
          this.updatePage();

        } else {
          for (let i = data.actions.length - 1; i >= 0; i--) {
            this.pageData.unshift({'episode_id': data.id, 'action_id': i, 'length': data.actions.length, ...data.actions[i]});
            this.pageData.pop();
          }
        }
      }
    });

    window.addEventListener('resize', this.resize);

    document.body.addEventListener('keydown', event => {
      if (document.activeElement !== document.body) return;
      if (event.metaKey || event.ctrlKey) return;

      if (event.code === 'ArrowLeft') {
        this.diffIndex(-1);
      } else if (event.code === 'ArrowRight') {
        this.diffIndex(1);
      } else if (event.code === 'ArrowUp') {
        this.diffIndex(-this.columnCount);
      } else if (event.code === 'ArrowDown') {
        this.diffIndex(this.columnCount);
      } else if (event.code === 'KeyA') {
        this.suffix = this.suffix.slice(0, 3) + 'after';
        localStorage.setItem('suffix', this.suffix);
      } else if (event.code === 'KeyE') {
        this.suffix = 'ed' + this.suffix.slice(2, 20);
        localStorage.setItem('suffix', this.suffix);
      } else if (event.code === 'KeyR') {
        this.suffix = 'rd' + this.suffix.slice(2, 20);
        localStorage.setItem('suffix', this.suffix);
      } else if (event.code === 'KeyC') {
        this.suffix = 'rc' + this.suffix.slice(2, 20);
        localStorage.setItem('suffix', this.suffix);
      } else if (event.code === 'KeyG') {
        this.suffix = this.suffix.slice(0, 3) + 'goal';
        localStorage.setItem('suffix', this.suffix);
      } else if (event.code === 'KeyV') {
        this.suffix = this.suffix.slice(0, 3) + 'v';
        localStorage.setItem('suffix', this.suffix);
      } else if (event.code === 'PageUp') {
        this.nextPage();
      } else if (event.code === 'PageDown') {
        this.prevPage();
      }
    });
  },
  methods: {
    updateActions: function(event) {
      localStorage.setItem('collection', this.collection);

      this.updatePage(0);
      this.updateStats();
    },
    updateDetail: function(episode_id, action_id) {
      this.currentEpisodeId = episode_id;
      this.currentActionId = action_id;

      return axios.get(`api/${this.collection}/${this.currentEpisodeId}`).then(response => {
        this.detailEpisode = response.data;
        this.detailAction = this.detailEpisode.actions[this.currentActionId];
      });
    },
    updatePage: function(detailIndex = null) {
      localStorage.setItem('column_count', this.columnCount);
      localStorage.setItem('show_box', this.showBox);
      localStorage.setItem('suffix', this.suffix);

      const params = {...this.filter, skip: this.getPageStartIndex(), limit: this.getPageLength()};
      return axios.get(`api/${this.collection}/actions`, { params }).then(response => {
        this.pageCount = Math.ceil(response.data.stats.length / this.getPageLength());
        this.pageData = response.data.actions;

        if (detailIndex != null) {
          if (detailIndex < 0) detailIndex = this.pageData.length + detailIndex;
          if (detailIndex < 0 || detailIndex >= this.pageData.length) return;

          const new_action = this.pageData[detailIndex];
          this.updateDetail(new_action.episode_id, new_action.action_id);
        }

        setTimeout(this.resize, 400);
      });
    },
    updateStats: function(event) {
      return axios.get(`api/${this.collection}/stats`, {params: {...this.filter}}).then(response => {
        this.stats = response.data;
      });
    },
    updateReward: function(event) {
      const newReward = this.detailAction.reward;
      return axios.post(`api/${this.collection}/${this.currentEpisodeId}/${this.currentActionId}/update-reward`, {reward: newReward}).then(response => {
        if (!response.data.success) return;

        this.pageData.find(e => e.episode_id == this.currentEpisodeId && e.action_id == this.currentActionId).reward = newReward;
      });
    },
    deleteEpisode: function() {
      return axios.post(`api/${this.collection}/${this.currentEpisodeId}/delete`).then(response => {
        if (!response.data.success) return;

        const index = this.pageData.findIndex(a => a.episode_id === this.currentEpisodeId);

        this.updatePage(index);
        this.updateStats();
      });
    },
    search: function(event) {
      this.updatePage();
      this.updateStats();
    },
    resize: function(event) {
      const viewpointHeight = document.documentElement.clientHeight - 50;
      const imageElement = document.getElementById('0');

      let imageHeight = 50;
      if (imageElement) {
        imageHeight = Math.max(imageElement.clientHeight, imageHeight);
      }

      const newRowCount = Math.max(Math.floor(viewpointHeight / imageHeight), 1);

      if (this.rowCount != newRowCount) {
        this.rowCount = newRowCount;
        localStorage.setItem('row_count', this.rowCount);

        this.updatePage();
      }
    },
    diffIndex: function(index_diff) {
      let index = this.pageData.findIndex(a => (a.episode_id === this.currentEpisodeId && a.action_id === this.currentActionId));
      if (index === -1) return;

      index += parseInt(index_diff);

      if ((this.pageIndex == 0 && index < 0) || (this.pageIndex === this.pageCount - 1 && index >= this.pageData.length)) return;

      if (index >= this.getPageLength()) {
        this.nextPage();
      } else if (index < 0) {
        this.prevPage();
      } else {
        const new_action = this.pageData[index];
        this.updateDetail(new_action.episode_id, new_action.action_id);
      }
    },
    copyToClipboard: function(text) {
      let testingCodeToCopy = document.querySelector('#copy-field');
      testingCodeToCopy.setAttribute('type', 'text');
      testingCodeToCopy.value = text;
      testingCodeToCopy.select();

      try {
        document.execCommand('copy');
        testingCodeToCopy.setAttribute('type', 'hidden');
      } catch (err) {
        alert('Oops, unable to copy');
      }
    },
    nextPage: function(event) {
      this.setPage(this.pageIndex + 1, 0);
    },
    prevPage: function(event) {
      this.setPage(this.pageIndex - 1, -1);
    },
    setPage: function(index, detailIndex = null) {
      if (index < 0 || index >= this.pageCount) return;

      this.pageIndex = index;
      // this.$router.push({name: '', query : { page: this.pageIndex } });
      this.updatePage(detailIndex);
    },
    getPageStartIndex: function() {
      return this.pageIndex * this.getPageLength();
    },
    getPageLength: function() {
      return this.rowCount * this.columnCount;
    },
  },
  mixins: [mixin],
});

const Live = Vue.extend({
  template: '#live-template',
  data () {
    return {
      lastAction: {},
    }
  },
  created () {
    this.collection = localStorage.getItem('collection') || this.collection;
    this.showBox = localStorage.getItem('show_box') || this.showBox;

    this.updateCollections().then(response => {
      this.updateLastAction();
    });

    this.socket.on('new-episode', data => {
      this.lastAction = data.actions.slice(-1)[0];
    });

    this.socket.on('new-attempt', data => {
      this.collection = data.collection;
      this.lastAction = data.action;
    });
  },
  methods: {
    updateLastAction: function(event) {
      localStorage.setItem('collection', this.collection);

      axios.get(`api/${this.collection}/episodes`, {params: {...this.filter, limit: 1}}).then(response => {
        axios.get(`api/${this.collection}/${response.data.episodes[0].id}`).then(response2 => {
          const action_id = response2.data.actions.length - 1;
          this.lastAction = response2.data.actions[action_id];
          this.lastAction.collection = response2.data.collection;
          this.lastAction.episode_id = response2.data.id;
          this.lastAction.action_id = action_id;
        });
      });
    }
  },
  mixins: [mixin],
});

// Get possible /database/, but not /live
const base = window.location.pathname.split('/').length > 2 ? window.location.pathname.split('/')[1] : '';

const router = new VueRouter({
  base,
  routes: [
    { path: '/', component: Overview },
    { path: '/live', component: Live },
  ],
  mode: 'history',
})

const app = new Vue({
  router
}).$mount('#app');
