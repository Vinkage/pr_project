
# Table of Contents

1.  [Creating a virtual machine](#org1677977)
    1.  [Google cloud](#org2ce269d)
        1.  [Making account and setting up billing](#orge672a7e):ATTACH:
        2.  [Requesting GPU quota and launching virtual machine](#org224df02):ATTACH:
    2.  [Azure student sponsorship](#org34cc87e)
2.  [Working with a server](#org5e6856d)
    1.  [Windows](#org02a6211)
        1.  [SSH](#org5ebc47d)
        2.  [tmux](#org7e8c79b)
        3.  [Google cloud](#orgb74b1e9)
    2.  [Mac (assumed everyone has a windows machine)](#orgf22c3ce)


<a id="org1677977"></a>

# Creating a virtual machine


<a id="org2ce269d"></a>

## Google cloud


<a id="orge672a7e"></a>

### Making account and setting up billing     :ATTACH:

1.  Make an account using your credit card info with this link <https://cloud.google.com/free>, if everything went alright you will be greeted by the google cloud home screen.

![img](/Users/mikevink/Dropbox (1)/org/.attach/8b/41ab28-71a9-4f20-9f64-c02b03455d2a/_20201224_123544Screenshot 2020-12-24 at 12.35.39.png)

1.  On the home screen there is a search bar, type: `billing`, it should show a screen similar to this one, which is the most important to us!:

![img](/Users/mikevink/Dropbox (1)/org/.attach/8b/41ab28-71a9-4f20-9f64-c02b03455d2a/_20201224_123827Screenshot 2020-12-24 at 12.38.21.png)

1.  In the bottom right you should see an option to upgrade your account to a paid account, we need this because otherwise we are locked out of requesting GPUs. Click it, and you should see a message reassuring that you will only start paying once your usage exceeds the 250 free trial credits.

2.  On the billing screen we also see health checks, click `View all health checks`. You should see this screen.
    
    ![img](/Users/mikevink/Dropbox (1)/org/.attach/8b/41ab28-71a9-4f20-9f64-c02b03455d2a/_20201224_124648Screenshot 2020-12-24 at 12.46.46.png)

3.  Click on create budget, you will need to give your budget a name and specify an amount. I gave the following `prproj` and `250`. Follow the default options for other steps. **In this way you will be alerted via email when your spending exceeds 50, 90 and 100 percent, but does NOT put a hard cap on your budget.**

4.  For hard caps and automatic instance stopping we need to do extra steps from <https://cloud.google.com/billing/docs/how-to/notify#functions_billing_auth-python>


<a id="org224df02"></a>

### Requesting GPU quota and launching virtual machine     :ATTACH:

1.  Now that we have our account ready, we can request GPUs. Type: `all quota` into the search bar. You should see a screen like this (note that I already did these steps):

![img](/Users/mikevink/Dropbox (1)/org/.attach/4a/e69b5b-f071-4ecf-aed0-fe34ffb41531/_20201224_130007Screenshot 2020-12-24 at 13.00.02.png)

1.  The important quota is `GPUs (all regions)` which is 0 by default. I requested 1 GPU by doing the following, but maybe we can do more. First use the `filter table` by clicking on the icon, and selecting metric from the drop down menu. Then type `gpus_all_regions.`

![img](/Users/mikevink/Dropbox (1)/org/.attach/4a/e69b5b-f071-4ecf-aed0-fe34ffb41531/_20201224_130339Screenshot 2020-12-24 at 13.03.34.png)

![img](/Users/mikevink/Dropbox (1)/org/.attach/4a/e69b5b-f071-4ecf-aed0-fe34ffb41531/_20201224_130400Screenshot 2020-12-24 at 13.03.58.png)

1.  In the GPUs (all regions) quota row click `ALL QUOTAS`,
    
    ![img](/Users/mikevink/Dropbox (1)/org/.attach/4a/e69b5b-f071-4ecf-aed0-fe34ffb41531/_20201224_130627Screenshot 2020-12-24 at 13.06.23.png)

tick the box, and then click `EDIT QUOTAS`,

![img](/Users/mikevink/Dropbox (1)/org/.attach/4a/e69b5b-f071-4ecf-aed0-fe34ffb41531/_20201224_130749Screenshot 2020-12-24 at 13.07.45.png)

this is the important screen, fill in the number of GPUs you want, fill in a request motivation (I put Utrecht University masters student), and click `NEXT`.

1.  Follow any steps, (can&rsquo;t remember) and if everything went alright you will receive an email within minutes saying your request has been approved.

2.  For more information on the following steps see: <https://cloud.google.com/ai-platform/deep-learning-vm/docs/pytorch_start_instance> and <https://cloud.google.com/compute/docs/gpus/gpu-regions-zones> (I now realised my mistake of making a GPU in asia-east, when the same is available in europe, I deleted the instance)

3.  The only way I could find to make a *preemptible* virtual machine was with the Cloud Console or the gcloud command line tool, using this link <https://cloud.google.com/ai-platform/deep-learning-vm/docs/pytorch_start_instance>. I assume you can use the gcloud command line tool here, more info could be found here <https://cloud.google.com/shell/docs/using-cloud-shell> or <https://cloud.google.com/sdk/docs/install>.

4.  The relevant code snippet that will **DEPLOY** a virtual machine right away is:
    
    <div class="ex" id="org5297d84">
    <p>
    export IMAGE<sub>FAMILY</sub>=&ldquo;pytorch-latest-gpu&rdquo;
    export ZONE=&ldquo;europe-west4-[abc]&rdquo;
    export INSTANCE<sub>NAME</sub>=&ldquo;my-instance&rdquo;
    </p>
    
    <p>
    gcloud compute instances create $INSTANCE<sub>NAME</sub> \
      &#x2013;zone=$ZONE \
      &#x2013;image-family=$IMAGE<sub>FAMILY</sub> \
      &#x2013;image-project=deeplearning-platform-release \
      &#x2013;maintenance-policy=TERMINATE \
      &#x2013;accelerator=&ldquo;type=nvidia-tesla-v100,count=1&rdquo; \
      &#x2013;metadata=&ldquo;install-nvidia-driver=True&rdquo; \
      &#x2013;preemptible
    </p>
    
    </div>
    
    You can change the `INSTANCE_NAME`, `IMAGE_FAMILY`, and `count` of gpus (if you have enough quota!).
    
    ![img](/Users/mikevink/Dropbox (1)/org/.attach/4a/e69b5b-f071-4ecf-aed0-fe34ffb41531/_20201224_132535Screenshot 2020-12-24 at 13.25.33.png)

**remember to stop the instance if you will not use it right away!!!**


<a id="org34cc87e"></a>

## Azure student sponsorship


<a id="org5e6856d"></a>

# Working with a server


<a id="org02a6211"></a>

## Windows


<a id="org5ebc47d"></a>

### SSH


<a id="org7e8c79b"></a>

### tmux


<a id="orgb74b1e9"></a>

### Google cloud


<a id="orgf22c3ce"></a>

## Mac (assumed everyone has a windows machine)

